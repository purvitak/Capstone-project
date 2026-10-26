"""Optional isolated-XGBoost chronology, physical units and checkpoint checks."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"pv_pipeline"))
AVAILABLE = importlib.util.find_spec("xgboost") is not None
if AVAILABLE:
    from pipelines.xgboost_nwp import run as R
    from pipelines.xgboost_nwp import training as T
    import xgboost as xgb


@unittest.skipUnless(AVAILABLE,"Install pipelines/xgboost_nwp/requirements.txt to test the optional backend")
class XGBoostTrainingTests(unittest.TestCase):
    def setUp(self):
        self.params = dict(next(R.configurations())[1],min_child_weight=2)

    def sample(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(20,4,3)).astype(np.float32)
        cs=np.full((20,4),600.)
        baseline=np.full_like(cs,300.)
        y=baseline+rng.uniform(-100,100,cs.shape)
        return x,y,cs,np.ones_like(cs,bool),baseline

    def test_residual_baseline_restoration_and_saved_model_roundtrip(self):
        x,y,cs,mask,baseline=self.sample();y[:]=350.
        model,info=T.fit_residual(x,y,cs,mask,baseline,self.params,max_rounds=3)
        pred=T.predict_kt(model,x,cs,baseline)
        np.testing.assert_allclose(pred*cs,350.,atol=1e-5)
        self.assertEqual(info["target"],"ghi_residual_to_nwp")
        self.assertAlmostEqual(info["inner_validation_rmse_wm2"],0.)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model.json";model.save_model(path)
            restored=xgb.Booster(model_file=path)
            np.testing.assert_array_equal(T.predict_kt(restored,x,cs,baseline),pred)

    def test_inner_purge_and_refit_use_only_their_chronological_targets(self):
        x,y,cs,mask,baseline=self.sample()
        y[15:]=999.
        recorded=[];original=T.xgb.train
        def capture(params,matrix,**kwargs):
            recorded.append(matrix.get_label().copy())
            return original(params,matrix,**kwargs)
        with patch.object(T.xgb,"train",side_effect=capture):
            model,info=T.fit_residual(x,y,cs,mask,baseline,self.params,max_rounds=3)
        np.testing.assert_array_equal(recorded[0],(y-baseline)[:15].ravel().astype(np.float32))
        np.testing.assert_array_equal(recorded[1],(y-baseline).ravel().astype(np.float32))
        self.assertEqual(info["inner_train_days"],15)
        self.assertEqual(info["inner_purge_days"],1)
        self.assertEqual(info["inner_validation_days"],4)
        self.assertEqual(model.num_boosted_rounds(),info["selected_iterations"])

    def test_iteration_selection_scores_clipped_ghi_not_index_error(self):
        x=np.ones((20,2,1),np.float32)
        cs=np.tile([100.,1000.],(20,1));y=.5*cs;baseline=np.zeros_like(cs)
        class Tree:
            def predict(self,matrix,iteration_range):
                return np.tile([60.,490.] if iteration_range[1]==1 else [50.,520.],matrix.num_row()//2)
        with patch.object(T.xgb,"train",return_value=Tree()) as factory:
            _,info=T.fit_residual(x,y,cs,np.ones_like(cs,bool),baseline,self.params,max_rounds=2)
        self.assertEqual(info["selected_iterations"],1)
        self.assertAlmostEqual(info["inner_validation_rmse_wm2"],10.)
        self.assertEqual(factory.call_args.kwargs["num_boost_round"],1)

    def test_night_targets_never_change_the_fitted_model(self):
        x,y,cs,mask,baseline=self.sample();mask[:,::2]=False
        other=y.copy();other[~mask]=1e8
        results=[]
        for target in (y,other):
            model,info=T.fit_residual(x,target,cs,mask,baseline,self.params,max_rounds=4)
            results.append((T.predict_kt(model,x,cs,baseline),info))
        np.testing.assert_array_equal(results[0][0],results[1][0])
        self.assertEqual(results[0][1],results[1][1])

    def test_invalid_baseline_alignment_and_empty_daylight_are_rejected(self):
        x,y,cs,mask,baseline=self.sample()
        with self.assertRaises(ValueError):T.fit_residual(x,y,cs,mask,baseline[:,:2],self.params,2)
        with self.assertRaises(ValueError):T.fit_residual(x,y,cs,np.zeros_like(mask),baseline,self.params,2)
        baseline[0,0]=np.nan
        with self.assertRaises(ValueError):T.fit_residual(x,y,cs,mask,baseline,self.params,2)

    def test_blend_fallback_is_whole_day_and_independent_of_weight(self):
        trial=np.full((3,4),.8);trial[1]=np.nan
        history=np.full_like(trial,.4);fallback=np.full_like(trial,.6)
        for weight in R.WEIGHTS:
            result=R.combine(trial,history,np.array([True,False,True]),fallback,weight)
            np.testing.assert_array_equal(result[1],fallback[1])
            np.testing.assert_allclose(result[0],weight*.8+(1-weight)*.4)

    def test_four_configurations_and_no_automatic_promotion_of_tradeoffs(self):
        self.assertEqual(len(dict(R.configurations())),4)
        ref={"rmse":100.,"mae":60.}
        rows=[{"candidate":"a","weight":.5,"rmse":90.,"mae":65.}]
        chosen,eligible=R.rank_tuning(rows,ref)
        self.assertFalse(eligible)
        self.assertEqual(chosen["candidate"],"a")
        rows.append({"candidate":"b","weight":.5,"rmse":95.,"mae":55.})
        chosen,eligible=R.rank_tuning(rows,ref)
        self.assertTrue(eligible)
        self.assertEqual(chosen["candidate"],"b")

    def test_checkpoint_rejects_xgboost_parameter_changes(self):
        data={"dates":np.array(["2024-01-01"]),"ghi":np.ones((1,4))}
        expected=R.contract("manifest","trial",self.params)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"oof.npz"
            R.L.save_predictions(path,np.ones((1,4)),data["dates"],expected)
            changed=R.contract("manifest","trial",dict(self.params,eta=.1))
            with self.assertRaisesRegex(ValueError,"settings differ"):
                R.E.load_checkpoint(path,data,changed)

    def test_budget_is_checked_during_training(self):
        budget=R.N.Budget(1)
        x,y,cs,mask,baseline=self.sample()
        with patch.object(R.N.time,"monotonic",return_value=budget.deadline+1):
            with self.assertRaises(R.N.BudgetExpired):
                T.fit_residual(x,y,cs,mask,baseline,self.params,max_rounds=3,budget=budget)


if __name__=="__main__":
    unittest.main()
