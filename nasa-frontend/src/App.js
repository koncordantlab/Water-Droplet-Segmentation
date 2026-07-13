import { useState, useRef, useEffect } from "react";
import "./App.css";
import { Box, LinearProgress } from "@mui/material";
import HomePage from "./pages/homepage/HomePage";
import NoCloseModal from "./components/noclosemodal/NoCloseModal";
import SummaryPage from "./pages/summarypage/SummaryPage";

const API_BASE = process.env.REACT_APP_BACKEND_API_URL || "/api";

const style = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 400,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
  borderRadius: 5
};

export default function App() {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  // Track Progress
  const [taskId, setTaskId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [eta, setEta] = useState(null);

  const [progressModal, setProgressModal] = useState(false);

  // results
  const [charts, setCharts] = useState(null);
  const [rows, setRows] = useState([]); // keep controlled as array
  const [overlaps, setOverlaps] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [executionTime, setExecutionTime] = useState(null);
  const [sizeDistribution, setSizeDistribution] = useState(null);

  // slider state
  const [sliderMax, setSliderMax] = useState(1);
  const [sliderValue, setSliderValue] = useState(1);

  // plot render tracking
  const [plotsToRender, setPlotsToRender] = useState(0);
  const [plotsRendered, setPlotsRendered] = useState(0);
  const plotRenderedIdsRef = useRef(new Set());
  const chartRenderStartRef = useRef(null);
  const [chartRenderElapsed, setChartRenderElapsed] = useState(null);

  // simple routing by pathname
  const pathname = typeof window !== "undefined" ? window.location.pathname : "/";

  useEffect(() => {
    const n = Math.max(1, (rows && rows.length) || 1);
    setSliderMax(n);
    setSliderValue(n);
  }, [rows]);

  const handleSubmit = async (ev, videoPath, isSegmentationOn, distInterval = 0, outputMode = "full", umPerPx = "") => {
    if (ev && ev.preventDefault) ev.preventDefault();

    setError(null);

    if (!videoPath) {
      setError("Please provide a video path.");
      return;
    }

    setLoading(true);
    setProgressModal(true);
    setProgress(0);
    setCharts(null);
    setRows([]);
    setOverlaps(null);
    setDownloadUrl(null);
    setExecutionTime(null);
    setSizeDistribution(null);
    try {
      const resp = await fetch(`${API_BASE}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_path: videoPath,
          save_overlay: isSegmentationOn,
          dist_interval: Number(distInterval) || 0,
          output_mode: outputMode,
          um_per_px: umPerPx === "" ? null : Number(umPerPx),
        }),
      });

      const data = await resp.json();
      if (data.status !== "ok") throw new Error(data.message || "Processing failed");

      setTaskId(data.task_id || null);
    } catch (err) {
      setError(err.message || "An error occurred during processing.");
      setLoading(false);
      setProgressModal(false);
    }
  };

  useEffect(() => {
  if (!taskId) return;

  const eventSource = new EventSource(`${API_BASE}/events/${taskId}`);

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      setProgress((p) => (data.progress > p ? data.progress : p));
      setEta((e) => data.eta || e);
      setMessage((m) => data.message || m);

      if (data.status === "completed") {
        setMessage(data.message || "Processing completed.");
        setCharts(data.charts || null);
        setRows(data.rows || []);
        setOverlaps(data.overlap_totals || null);
        setDownloadUrl(data.download_url || data.excel_path || null);
        setExecutionTime(data.execution_time || null);
        setSizeDistribution(data.size_distribution || null);
        setLoading(false);
        setProgressModal(false);

        // prepare plot render tracking
        const ids = [];
        if (data.charts?.pct) ids.push("pct");
        if (data.charts?.ov) ids.push("ov");
        if (data.charts?.donuts)
          ids.push("donut-water", "donut-ice", "donut-void", "donut-conf");
        setPlotsToRender(ids.length);
        setPlotsRendered(0);
        plotRenderedIdsRef.current.clear();

        if (ids.length > 0) {
          chartRenderStartRef.current = performance.now();
          setChartRenderElapsed(null);
        } else {
          setChartRenderElapsed(0);
        }

        // navigate to summary
        window.history.pushState({}, "", "/summary");
        setTimeout(() => window.scrollTo(0, 0), 80);
        eventSource.close();
      }
    } catch (err) {
      eventSource.close();
    }
  };

  eventSource.onerror = (err) => {
    setLoading(false);
    setProgressModal(false);
    eventSource.close();
  };

  return () => {
    eventSource.close();
  };
}, [taskId]);


  const markPlotRendered = (id) => {
    if (!plotRenderedIdsRef.current.has(id)) {
      plotRenderedIdsRef.current.add(id);
      setPlotsRendered((prev) => {
        const next = prev + 1;
        if (plotsToRender > 0 && next === plotsToRender && chartRenderStartRef.current) {
          const now = performance ? performance.now() : Date.now();
          setChartRenderElapsed(Math.round(now - chartRenderStartRef.current));
          chartRenderStartRef.current = null;
        }
        return next;
      });
    }
  };

  return (
    <>
      {loading && (
        <NoCloseModal
          openModal={progressModal}
        >
          <Box sx={style}>
            <Box sx={{width: '100%'}}>
              <LinearProgress variant="determinate" value={progress} sx={{height: "1rem", borderRadius: 5, transition: "all 0.8s ease-out"}} />
              
              {message && (
                <div style={{ marginTop: 12 }}>
                  {message}
                </div>
              )}

              {eta != null && (
                <div style={{ marginTop: 6 }}>
                  Estimated time remaining: {Math.round(eta)} seconds
                </div>
              )}
            </Box>
          </Box>
        </NoCloseModal>
      )}
      { pathname === "/summary" ?
        <SummaryPage
          rows={rows}
          overlaps={overlaps}
          markPlotRendered={markPlotRendered}
          chart={charts}
          sliderMax={sliderMax}
          sliderValue={sliderValue}
          chartRenderElapsed={chartRenderElapsed}
          plotsToRender={plotsToRender}
          executionTime={executionTime}
          setSliderValue={setSliderValue}
          plotsRendered={plotsRendered}
          sizeDistribution={sizeDistribution}
        /> : <HomePage loading={loading} handleSubmit={handleSubmit} downloadUrl={downloadUrl} error={error} />}
    </>
  );
}