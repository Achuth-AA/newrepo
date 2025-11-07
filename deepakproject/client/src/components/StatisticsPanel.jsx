import React from 'react';

const StatisticsPanel = ({ stats, isLoading }) => {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-gray-800 rounded-lg p-6 shadow-lg animate-pulse">
            <div className="h-4 bg-gray-700 rounded w-3/4 mb-4"></div>
            <div className="h-8 bg-gray-700 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    );
  }

  const cards = [
    {
      title: 'Total Readings',
      value: stats.total_readings.toLocaleString(),
      icon: '📊',
      color: 'blue',
    },
    {
      title: 'Anomalies Detected',
      value: stats.anomaly_count.toLocaleString(),
      icon: '⚠️',
      color: 'red',
    },
    {
      title: 'Anomaly Rate',
      value: `${stats.anomaly_percentage.toFixed(2)}%`,
      icon: '📈',
      color: 'yellow',
    },
    {
      title: 'Active Sensors',
      value: Object.keys(stats.sensor_health).length,
      icon: '🌡️',
      color: 'green',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map((card, index) => (
        <div
          key={index}
          className="bg-gray-800 rounded-lg p-6 shadow-lg border border-gray-700 hover:border-gray-600 transition-colors"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-gray-400 text-sm font-medium">{card.title}</h3>
            <span className="text-2xl">{card.icon}</span>
          </div>
          <div className={`text-3xl font-bold text-${card.color}-400`}>
            {card.value}
          </div>
        </div>
      ))}
    </div>
  );
};

export default StatisticsPanel;
