import Plot from "react-plotly.js"
import { buildPercentageFigure } from "../../../utils/chartsUtils"

const PercentageChart = ({chart, markPlotRendered, sliderValue}) => {

    const fig = buildPercentageFigure(chart, sliderValue)

    const handleInitialized = () => {
        markPlotRendered("pct");
    }
    
    const handleUpdate = () => {
        markPlotRendered("pct");
    }

    if(!fig) return null;

    return (
        <Plot 
            data = {fig.data}
            layout = {fig.layout}
            config = {fig.config}
            key={`pct`}
            style={{ width: "100%", height: 360 }}
            onInitialized={handleInitialized}
            onUpdate={handleUpdate}
        />
    )
}

export default PercentageChart
