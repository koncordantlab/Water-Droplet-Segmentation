import { useEffect, useState } from "react";

import "./slider.css"

const Slider = ({ max, step, defaultValue, updateValue, styles }) => {
  const [value, setValue] = useState(defaultValue);
  const [isActive, setIsActive] = useState(false);

  const handleCommitChange = (e) => {
      updateValue(Number(e.target.value));
  }

  const handleChange = (e) => {
    setValue(Math.max(1, Math.min(Number(e.target.value || 1), max)))
  }

  const percentage = ((value - 1) / (max -1)) * 100;

  useEffect(() => {
    setValue(defaultValue)
  }, [defaultValue]);

  return (
    <div className="slider-container">
      <div className="slider-wrapper">
        <input
          type="range"
          min={1}
          max={max}
          step={step}
          value={value}
          className="slider-input"
          onChange={handleChange}
          onMouseUp={handleCommitChange}
          onTouchEnd={handleCommitChange}
          onMouseEnter={() => setIsActive(true)}
          onMouseLeave={() => setIsActive(false)}
          style={{...styles}}
        />
        {
          isActive && (
            <div
              className="slider-tooltip"
              style={{
                left: `calc(${percentage}% + (${8-(percentage/100) * 18}px))`
              }}
            > {value} </div>
          )
        }
      </div>
    </div>
  )
}

export default Slider;
