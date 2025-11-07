import React, { useState } from 'react';

const SensorGrid = ({ sensorHealth }) => {
  const [selectedSensor, setSelectedSensor] = useState(null);
  const [filter, setFilter] = useState('all');

  // Generate sensor cards
  const totalSensors = 54;
  const sensors = Array.from({ length: totalSensors }, (_, i) => {
    const sensorId = i + 1;
    const health = sensorHealth[sensorId] || 'good';
    return { id: sensorId, health };
  });

  // Filter sensors
  const filteredSensors = sensors.filter((sensor) => {
    if (filter === 'all') return true;
    return sensor.health === filter;
  });

  const getHealthColor = (health) => {
    switch (health) {
      case 'good':
        return 'bg-green-500 border-green-400';
      case 'warning':
        return 'bg-yellow-500 border-yellow-400';
      case 'critical':
        return 'bg-red-500 border-red-400';
      default:
        return 'bg-gray-500 border-gray-400';
    }
  };

  const getHealthIcon = (health) => {
    switch (health) {
      case 'good':
        return '✓';
      case 'warning':
        return '⚠';
      case 'critical':
        return '✕';
      default:
        return '?';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg border border-gray-700">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-xl font-bold text-white mb-1">Sensor Health Grid</h3>
          <p className="text-gray-400 text-sm">
            {filteredSensors.length} of {totalSensors} sensors shown
          </p>
        </div>
        <div className="flex space-x-2">
          {['all', 'good', 'warning', 'critical'].map((status) => (
            <button
              key={status}
              onClick={() => setFilter(status)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                filter === status
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Sensor Grid */}
      <div className="grid grid-cols-6 sm:grid-cols-9 md:grid-cols-12 lg:grid-cols-18 gap-3">
        {filteredSensors.map((sensor) => (
          <div
            key={sensor.id}
            onClick={() => setSelectedSensor(sensor)}
            className={`${getHealthColor(
              sensor.health
            )} aspect-square rounded-lg border-2 cursor-pointer hover:scale-110 transition-transform flex items-center justify-center relative group`}
          >
            <span className="text-white font-bold text-sm">{sensor.id}</span>

            {/* Tooltip */}
            <div className="hidden group-hover:block absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded whitespace-nowrap z-10 shadow-lg">
              <div className="font-semibold">Sensor {sensor.id}</div>
              <div className="text-gray-300">
                Status: {sensor.health.toUpperCase()}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex justify-center space-x-6 mt-6">
        <div className="flex items-center">
          <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
          <span className="text-gray-300 text-sm">Good</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 bg-yellow-500 rounded mr-2"></div>
          <span className="text-gray-300 text-sm">Warning</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
          <span className="text-gray-300 text-sm">Critical</span>
        </div>
      </div>

      {/* Selected Sensor Detail */}
      {selectedSensor && (
        <div className="mt-6 p-4 bg-gray-700 rounded-lg border border-gray-600">
          <div className="flex justify-between items-center">
            <div>
              <h4 className="text-lg font-bold text-white">
                Sensor {selectedSensor.id}
              </h4>
              <p className="text-gray-400 text-sm mt-1">
                Health Status: <span className="font-semibold text-white">{selectedSensor.health.toUpperCase()}</span>
              </p>
            </div>
            <button
              onClick={() => setSelectedSensor(null)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SensorGrid;
