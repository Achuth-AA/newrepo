import React, { useState, useEffect } from 'react';

const RealTimeMonitor = ({ selectedModel }) => {
  const [recentAnomalies, setRecentAnomalies] = useState([]);
  const [isMonitoring, setIsMonitoring] = useState(false);

  // Simulate real-time data for demonstration
  useEffect(() => {
    if (isMonitoring) {
      const interval = setInterval(() => {
        const mockAnomaly = {
          timestamp: new Date().toISOString(),
          moteid: Math.floor(Math.random() * 54) + 1,
          temperature: (Math.random() * 10 + 20).toFixed(2),
          humidity: (Math.random() * 30 + 30).toFixed(2),
          is_anomaly: Math.random() > 0.8,
          confidence: (Math.random() * 0.5 + 0.5).toFixed(2),
        };

        if (mockAnomaly.is_anomaly) {
          setRecentAnomalies((prev) => [mockAnomaly, ...prev.slice(0, 9)]);
        }
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [isMonitoring]);

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg border border-gray-700">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-xl font-bold text-white mb-1">
            Real-Time Monitoring
          </h3>
          <p className="text-gray-400 text-sm">
            Model: {selectedModel.toUpperCase()}
          </p>
        </div>
        <button
          onClick={() => setIsMonitoring(!isMonitoring)}
          className={`px-6 py-2 rounded-lg font-medium transition-colors ${
            isMonitoring
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-green-600 hover:bg-green-700 text-white'
          }`}
        >
          {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
        </button>
      </div>

      {/* Status Indicator */}
      <div className="mb-6 flex items-center space-x-2">
        <div
          className={`h-3 w-3 rounded-full ${
            isMonitoring ? 'bg-green-500 animate-pulse' : 'bg-gray-600'
          }`}
        ></div>
        <span className="text-gray-300 text-sm">
          {isMonitoring ? 'Monitoring Active' : 'Monitoring Paused'}
        </span>
      </div>

      {/* Recent Anomalies List */}
      <div className="space-y-3">
        <h4 className="text-lg font-semibold text-white mb-4">
          Recent Anomalies ({recentAnomalies.length})
        </h4>
        {recentAnomalies.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            {isMonitoring
              ? 'Monitoring for anomalies...'
              : 'Start monitoring to detect anomalies'}
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {recentAnomalies.map((anomaly, index) => (
              <div
                key={index}
                className="bg-gray-700 rounded-lg p-4 border border-red-500/30"
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <span className="text-red-400 font-semibold">
                      Sensor {anomaly.moteid}
                    </span>
                    <span className="text-gray-400 text-sm ml-2">
                      {new Date(anomaly.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <span className="bg-red-600 text-white text-xs px-2 py-1 rounded">
                    ANOMALY
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Temp:</span>
                    <span className="text-white ml-1">{anomaly.temperature}°C</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Humidity:</span>
                    <span className="text-white ml-1">{anomaly.humidity}%</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Confidence:</span>
                    <span className="text-white ml-1">{(anomaly.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default RealTimeMonitor;
