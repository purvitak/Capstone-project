"""Browser smoke test against running local services; requires Playwright + Chrome.

The 10 kWp / 0.8 values below are test inputs, not product defaults or site claims.
"""
from pathlib import Path
import argparse
import csv
import io
import json
import re

from playwright.sync_api import sync_playwright, expect

HERE=Path(__file__).resolve().parent


def ready(page):
    page.wait_for_timeout(250)
    page.wait_for_function("document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state') === 'notRunning'",timeout=30000)
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def download_rows(page):
    with page.expect_download() as waiting:
        page.get_by_role("button",name=re.compile("Download this view")).click()
    item=waiting.value
    return list(csv.DictReader(io.StringIO(Path(item.path()).read_text(encoding="utf-8-sig"))))


def screenshot(page,name):
    page.locator('[data-testid="stMain"]').evaluate('(el)=>el.scrollTo(0,0)')
    page.screenshot(path=str(HERE/'assets'/name),full_page=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="http://127.0.0.1:8501")
    parser.add_argument("--chrome",default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    parser.add_argument("--inspect-only",action="store_true")
    args=parser.parse_args()
    checks={}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=args.chrome,headless=True)
        page=browser.new_page(viewport={"width":1512,"height":1400},device_scale_factor=1)
        errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto(args.url,wait_until="networkidle")
        page.get_by_text("Solar Energy Forecast Dashboard",exact=True).wait_for(timeout=30000)
        ready(page)
        if args.inspect_only:
            print(page.locator('body').inner_text()[:6000])
            print("Inputs",page.locator('input').evaluate_all('(xs)=>xs.map(x=>({type:x.type,value:x.value,placeholder:x.placeholder}))'))
            print("Chart ancestors",page.locator('[data-testid="stPlotlyChart"]').first.evaluate('(x)=>{let a=[];for(let i=0;x&&i<6;i++,x=x.parentElement)a.push({tag:x.tagName,id:x.dataset.testid,style:x.getAttribute("style"),class:x.className});return a}'))
            screenshot(page,'preview-desktop.png')
            browser.close();return
        expect(page.get_by_text("API connected · Saved forecast",exact=True)).to_be_visible()
        checks["real_api_connection"]=True
        expect(page.get_by_text("Add your PV system",exact=True)).to_be_visible()
        checks["pv_parameters_initially_empty"]=True
        date_input=page.locator('[data-testid="stDateInput"] input')
        date_input.fill("2026-02-20");date_input.press("Enter")
        page.get_by_role("button",name=re.compile("^Friday, 20 February 2026")).click()
        ready(page)
        expect(page.get_by_text("Friday, 20 February 2026",exact=True)).to_be_visible()
        checks["date_selection"]=True
        rows=download_rows(page)
        assert len(rows)==144 and "pv_power_kw" not in rows[0]
        checks["daily_csv_144_real_ghi_intervals"]=True
        numbers=page.locator('[data-testid="stNumberInput"] input')
        numbers.nth(0).fill("10");numbers.nth(0).press("Enter");ready(page)
        numbers.nth(1).fill("0.8");numbers.nth(1).press("Enter");ready(page)
        expect(page.get_by_text("Add your PV system",exact=True)).to_have_count(0)
        assert page.locator('[data-testid="stPlotlyChart"]').count()==3
        rows=download_rows(page)
        assert len(rows)==144 and "pv_power_kw" in rows[0]
        checks["user_entered_pv_and_energy"]=True
        screenshot(page,'preview-desktop.png')
        bars=page.locator('[data-testid="stPlotlyChart"]').nth(2).locator('.barlayer .point path')
        # Plotly's transparent interaction rectangle intentionally overlays bars.
        # A real pointer click at the bar's centre must go through that rectangle.
        bars.nth(12).click(force=True);ready(page)
        expect(page.get_by_text("Hour · IST",exact=True)).to_be_visible()
        assert len(download_rows(page))==6
        checks["hourly_plot_click_drilldown"]=True
        inspector=page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text("Select interval",exact=True))
        inspector.get_by_role("combobox").click()
        page.get_by_role("option",name="12:20 IST",exact=True).click();ready(page)
        expect(page.get_by_text("10-minute period",exact=True)).to_be_visible()
        assert len(download_rows(page))==1
        checks["manual_interval_drilldown_and_filtered_csv"]=True
        screenshot(page,'preview-interval.png')
        page.get_by_role("button",name="← Day view",exact=True).click();ready(page)
        assert len(download_rows(page))==144
        checks["back_navigation"]=True
        page.get_by_role("button",name="Clear PV settings",exact=True).click();ready(page)
        expect(page.get_by_text("Add your PV system",exact=True)).to_be_visible()
        checks["clear_pv_restores_unavailable_values"]=True
        page.set_viewport_size({"width":390,"height":844});ready(page)
        collapse=page.locator('[data-testid="stSidebarCollapseButton"] button')
        bounds=collapse.bounding_box() if collapse.count() else None
        if bounds and 0<=bounds['x'] and bounds['x']+bounds['width']<=390:
            collapse.click();page.wait_for_timeout(250)
        opener=page.locator('[data-testid="stExpandSidebarButton"]')
        expect(opener).to_be_visible()
        opener.click();page.wait_for_timeout(350)
        expect(page.get_by_label("Installed DC capacity (kWp)",exact=True)).to_be_visible()
        page.locator('[data-testid="stSidebarCollapseButton"] button').click();page.wait_for_timeout(350)
        checks["mobile_pv_settings_accessible"]=True
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2')
        screenshot(page,'preview-mobile.png')
        checks["mobile_no_horizontal_overflow"]=True
        checks["browser_uncaught_errors"]=errors
        assert not errors,errors
        browser.close()
    (HERE/'browser_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
