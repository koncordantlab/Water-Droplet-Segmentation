import computeDynamicSummary from "../../utils/dynamicSummary"
import PercentageChart from "./summaryVisuals/PercentageChart";
import OverlapChart from "./summaryVisuals/OverlapChart";
import Slider from "../../components/slider/Slider";
import PieCharts from "./summaryVisuals/PieCharts";
import SizeDistribution from "./summaryVisuals/SizeDistribution";

import "./summaryPage.css"

const SummaryPage = ({
    rows,
    overlaps,
    markPlotRendered,
    chart,
    sliderMax,
    sliderValue,
    setSliderValue,
    chartRenderElapsed,
    plotsRendered,
    plotsToRender,
    executionTime,
    sizeDistribution
}) => {
    const dynamicText = computeDynamicSummary(rows, overlaps);

    const handleGoBack = () => {
        window.history.pushState({}, '', '/');
        window.location.reload();
    }
  return (
    <div className="summary-page-container">
        <button 
            className="go-back-button"
            onClick={handleGoBack}
        >
            Back to Home
        </button>
        
        <div className="charts-container">
            <div className="left-section">
                <h4 style={{textAlign: "center"}}>Water (%) & Ice (%)</h4>
                <div>
                    <PercentageChart 
                        chart={chart}
                        markPlotRendered={markPlotRendered}
                        sliderValue={sliderValue}
                    />
                </div>
                <ul style={{ marginLeft: 18, marginTop: 8 }}>
                    <li>Blue = Water (%)</li>
                    <li>Red = Ice (%)</li>
                </ul>
            </div>
            <div className="right-section">
                <h4 style={{textAlign: "center"}}>Overlap Counts</h4>
                <div>
                    <OverlapChart 
                        chart={chart}
                        markPlotRendered={markPlotRendered}
                        sliderValue={sliderValue}
                    />
                </div>
                <ul style={{ marginLeft: 18, marginTop: 8 }}>
                    <li>Blue = Water–Water</li>
                    <li>Red = Ice–Ice</li>
                    <li>Green = Water–Ice</li>
                </ul>
            </div>
        </div>

        <div style={{ textAlign: "center", marginTop: 20 }}>
            <div id="slider-output-container">Showing first {sliderValue} of {rows.length || 0} processed frames</div>
            <Slider 
                max={sliderMax}
                step={1}
                defaultValue={sliderValue}
                updateValue={(val) => setSliderValue(val)}
                styles={{
                    marginTop: 20
                }}
            />
        </div>

        <div id="dynamic-summary-p" style={{ width: "80%", margin: "2rem auto", textAlign: "center", fontSize: 18, fontStyle: "italic", color: "#e6c645" }}>
          {dynamicText || "Summary not available."}
        </div>

        <div style={{ width: "95%", margin: "2rem auto", textAlign: "center" }}>
          <PieCharts chart={chart} markPlotRendered={markPlotRendered}/>
        </div>

        <SizeDistribution sizeDistribution={sizeDistribution} />

        <div id="overlap-summary" style={{ width: "95%", margin: "2rem auto", fontSize: 18 }}>
          <h4 style={{ textAlign: "center" }}>Overlap Summary</h4>
          <ul style={{ listStyle: "none", padding: 0, textAlign: "center" }}>
            <li>Water–Water total: {overlaps?.ww ?? 0}</li>
            <li>Ice–Ice total: {overlaps?.ii ?? 0}</li>
            <li>Water–Ice total: {overlaps?.mixed ?? 0}</li>
          </ul>
        </div>

        <div style={{ width: "95%", margin: "2rem auto" }}>
          <h3>Raw rows</h3>
          <pre style={{ maxHeight: 300, overflow: "auto", border: "1px solid #ddd", padding: 8, color: "#222", backgroundColor: "#fff" }}>{JSON.stringify((rows || []).slice(0, sliderValue), null, 2)}</pre>
        </div>

        <div style={{ marginTop: 8 }}>
          {chartRenderElapsed == null ? <span style={{ color: "#ccc" }}>Rendering charts... ({plotsRendered}/{plotsToRender})</span> : <span style={{ color: "#e6c645" }}>Charts rendered in {chartRenderElapsed} ms</span>}
        </div>
        {executionTime != null && (
          <div style={{ marginTop: 6, color: "#ccc" }}>Backend execution: {executionTime} seconds</div>
        )}
    </div>
  )
}

export default SummaryPage
