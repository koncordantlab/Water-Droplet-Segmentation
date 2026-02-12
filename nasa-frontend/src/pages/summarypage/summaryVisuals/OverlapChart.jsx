import Plot from "react-plotly.js"
import { buildOverlappingFigure } from "../../../utils/chartsUtils"

const OverlapChart = ({ chart, markPlotRendered, sliderValue }) => {

    const fig = buildOverlappingFigure(chart, sliderValue)

    const handleInitialized = () => {
        markPlotRendered("ov")
    }

    const handleUpdate = () => {
        markPlotRendered("ov")
    }

    if (!fig) return null;

    return (
        <Plot 
            data={fig.data}
            layout={fig.layout}
            config={fig.config}
            key={`ov`}
            style={{width: "100%", height: 360}}
            onInitialized={handleInitialized}
            onUpdate={handleUpdate}
        />
    )
}

export default OverlapChart
