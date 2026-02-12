import Plot from "react-plotly.js";
import { donutFigure } from "../../../utils/chartsUtils";

const PieCharts = ({chart, markPlotRendered}) => {
    if(!chart?.donuts) return null;
    const donuts = chart.donuts;

  return (
    <div style={{ display: "flex", gap: 12, justifyContent: "space-between" }}>
        {donuts.water_count && donuts.water_count > 0 ? (
            <div style={{width: "24%"}}>
                          <Plot key={`donut-water`} {...donutFigure(["Water"], [donuts.water_count], "Water Count")} style={{ width: "100%", height: 220 }} onInitialized={() => markPlotRendered("donut-water")} onUpdate={() => markPlotRendered("donut-water")} />
            </div>
        ) : (
          <Plot key={`donut-water`} {...donutFigure(["Water"], [donuts.water_count], "Water Count")} style={{ width: "100%", height: 220, display: "none" }} onInitialized={() => markPlotRendered("donut-water")} onUpdate={() => markPlotRendered("donut-water")} />
        )}

        {donuts.ice_count && donuts.ice_count > 0 ? (
            <div style={{ width: "24%" }}>
                      <Plot key={`donut-ice`} {...donutFigure(["Ice"], [donuts.ice_count], "Ice Count")} style={{ width: "100%", height: 220 }} onInitialized={() => markPlotRendered("donut-ice")} onUpdate={() => markPlotRendered("donut-ice")} />
                    </div>
        ) : (
          <Plot key={`donut-ice`} {...donutFigure(["Ice"], [donuts.ice_count], "Ice Count")} style={{ width: "100%", height: 220, display: "none" }} onInitialized={() => markPlotRendered("donut-ice")} onUpdate={() => markPlotRendered("donut-ice")} />
        )}

        <div style={{ width: "24%" }}>
            <Plot key={`donut-void`} {...donutFigure(["Void", "Rest"], [donuts.void_pct_avg, Math.max(0, 100 - donuts.void_pct_avg)], "Void (%)")} style={{ width: "100%", height: 220 }} onInitialized={() => markPlotRendered("donut-void")} onUpdate={() => markPlotRendered("donut-void")} />
        </div>

        <div style={{ width: "24%" }}>
            <Plot key={`donut-conf`} {...donutFigure(["Avg Conf", "Rest"], [donuts.avg_conf, Math.max(0, 100 - donuts.avg_conf)], "Avg Confidence (%)")} style={{ width: "100%", height: 220 }} onInitialized={() => markPlotRendered("donut-conf")} onUpdate={() => markPlotRendered("donut-conf")} />
        </div>
    </div>
  )
}

export default PieCharts
