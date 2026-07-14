import { useState } from "react";
import Plot from "react-plotly.js";
import { buildSizeHistogramFigure } from "../../../utils/chartsUtils";

const WATER_COLOR = "#1f77b4";
const ICE_COLOR = "#d62728";

const fmt = (v) => (v == null ? "—" : Number(v).toFixed(2));

const StatsTable = ({ water, ice }) => (
    <table style={{ width: "100%", borderCollapse: "collapse", color: "#fff", fontSize: 14 }}>
        <thead>
            <tr style={{ borderBottom: "1px solid #555" }}>
                <th style={{ textAlign: "left", padding: "6px 10px" }}>Class</th>
                <th style={{ textAlign: "right", padding: "6px 10px" }}>Count</th>
                <th style={{ textAlign: "right", padding: "6px 10px" }}>Min</th>
                <th style={{ textAlign: "right", padding: "6px 10px" }}>Max</th>
                <th style={{ textAlign: "right", padding: "6px 10px" }}>Mean</th>
                <th style={{ textAlign: "right", padding: "6px 10px" }}>Median</th>
                <th style={{ textAlign: "right", padding: "6px 10px" }}>Std</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style={{ padding: "6px 10px", color: WATER_COLOR }}>Water</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{water?.count ?? 0}</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{fmt(water?.stats?.min)}</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{fmt(water?.stats?.max)}</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{fmt(water?.stats?.mean)}</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{fmt(water?.stats?.median)}</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{fmt(water?.stats?.std)}</td>
            </tr>
            <tr>
                <td style={{ padding: "6px 10px", color: ICE_COLOR }}>Ice</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{ice?.count ?? 0}</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{fmt(ice?.stats?.min)}</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{fmt(ice?.stats?.max)}</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{fmt(ice?.stats?.mean)}</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{fmt(ice?.stats?.median)}</td>
                <td style={{ textAlign: "right", padding: "6px 10px" }}>{fmt(ice?.stats?.std)}</td>
            </tr>
        </tbody>
    </table>
);

const SizeDistribution = ({ sizeDistribution }) => {
    const [selectedIdx, setSelectedIdx] = useState(0);

    if (!sizeDistribution || !Array.isArray(sizeDistribution.checkpoints) || sizeDistribution.checkpoints.length === 0) {
        return null;
    }

    const checkpoints = sizeDistribution.checkpoints;
    const idx = Math.min(selectedIdx, checkpoints.length - 1);
    const checkpoint = checkpoints[idx];

    const waterYMax = sizeDistribution.y_max?.water ?? 0;
    const iceYMax = sizeDistribution.y_max?.ice ?? 0;
    const waterFig = buildSizeHistogramFigure(checkpoint.water, WATER_COLOR, "Water — size distribution", waterYMax);
    const iceFig = buildSizeHistogramFigure(checkpoint.ice, ICE_COLOR, "Ice — size distribution", iceYMax);

    return (
        <div style={{ width: "95%", margin: "2rem auto" }}>
            <h3 style={{ textAlign: "center" }}>Droplet Size Distribution</h3>
            <div style={{ textAlign: "center", color: "#ccc", marginBottom: 12 }}>
                Sampled every {sizeDistribution.interval} processed frame(s) — unit: {sizeDistribution.unit}
            </div>

            <div style={{ textAlign: "center", marginBottom: 16 }}>
                <label htmlFor="size-dist-frame-select" style={{ marginRight: 8, color: "#fff" }}>
                    Checkpoint frame:
                </label>
                <select
                    id="size-dist-frame-select"
                    value={idx}
                    onChange={(e) => setSelectedIdx(parseInt(e.target.value, 10) || 0)}
                    style={{ padding: "4px 8px", fontSize: 14 }}
                >
                    {checkpoints.map((cp, i) => (
                        <option key={cp.frame} value={i}>
                            Frame {cp.frame}{i === checkpoints.length - 1 && cp.frame % sizeDistribution.interval !== 0 ? " (final)" : ""}
                        </option>
                    ))}
                </select>
            </div>

            <div style={{ marginBottom: 16 }}>
                <StatsTable water={checkpoint.water} ice={checkpoint.ice} />
            </div>

            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
                <div style={{ flex: "1 1 400px", minWidth: 320 }}>
                    {waterFig && (
                        <Plot
                            data={waterFig.data}
                            layout={waterFig.layout}
                            config={waterFig.config}
                            key={`water-hist-${checkpoint.frame}`}
                            style={{ width: "100%", height: 320 }}
                        />
                    )}
                </div>
                <div style={{ flex: "1 1 400px", minWidth: 320 }}>
                    {iceFig && (
                        <Plot
                            data={iceFig.data}
                            layout={iceFig.layout}
                            config={iceFig.config}
                            key={`ice-hist-${checkpoint.frame}`}
                            style={{ width: "100%", height: 320 }}
                        />
                    )}
                </div>
            </div>
        </div>
    );
};

export default SizeDistribution;
