import React, { useState, useEffect } from 'react';

const AnomalyChart = () => {
  const [timeRange, setTimeRange] = useState('1h');
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    // Generate mock time-series data
    const generateData = () => {
      const data = [];
      const now = new Date();
      const points = 20;

      for (let i = points; i >= 0; i--) {
        const timestamp = new Date(now - i * 3 * 60 * 1000); // 3 minutes intervals
        data.push({
          time: timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          normal: Math.floor(Math.random() * 50 + 150),
          anomaly: Math.floor(Math.random() * 20 + 5),
        });
      }
      return data;
    };

    setChartData(generateData());
  }, [timeRange]);

  const maxValue = Math.max(...chartData.map((d) => d.normal + d.anomaly));

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg border border-gray-700">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-xl font-bold text-white mb-1">
            Anomaly Detection Timeline
          </h3>
          <p className="text-gray-400 text-sm">
            Real-time detection over time
          </p>
        </div>
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className="bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="1h">Last Hour</option>
          <option value="6h">Last 6 Hours</option>
          <option value="24h">Last 24 Hours</option>
          <option value="7d">Last 7 Days</option>
        </select>
      </div>

      {/* Chart Area */}
      <div className="relative h-64">
        <div className="absolute inset-0 flex items-end justify-between space-x-1">
          {chartData.map((data, index) => (
            <div key={index} className="flex-1 flex flex-col items-center">
              <div className="w-full flex flex-col justify-end items-center space-y-1 h-48">
                {/* Anomaly bar */}
                <div
                  className="w-full bg-red-500 rounded-t transition-all hover:bg-red-400 cursor-pointer group relative"
                  style={{
                    height: `${(data.anomaly / maxValue) * 100}%`,
                    minHeight: data.anomaly > 0 ? '4px' : '0px',
                  }}
                >
                  <div className="hidden group-hover:block absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap">
                    Anomalies: {data.anomaly}
                  </div>
                </div>
                {/* Normal bar */}
                <div
                  className="w-full bg-green-500 rounded-t transition-all hover:bg-green-400 cursor-pointer group relative"
                  style={{
                    height: `${(data.normal / maxValue) * 100}%`,
                    minHeight: '4px',
                  }}
                >
                  <div className="hidden group-hover:block absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap">
                    Normal: {data.normal}
                  </div>
                </div>
              </div>
              {index % 4 === 0 && (
                <span className="text-xs text-gray-400 mt-2 transform -rotate-45 origin-top-left">
                  {data.time}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex justify-center space-x-6 mt-8">
        <div className="flex items-center">
          <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
          <span className="text-gray-300 text-sm">Normal Readings</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
          <span className="text-gray-300 text-sm">Anomalies Detected</span>
        </div>
      </div>
    </div>
  );
};

export default AnomalyChart;
