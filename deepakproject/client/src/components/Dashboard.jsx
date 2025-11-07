import React, { useState, useEffect } from 'react';
import StatisticsPanel from './StatisticsPanel';
import RealTimeMonitor from './RealTimeMonitor';
import AnomalyChart from './AnomalyChart';
import SensorGrid from './SensorGrid';
import ModelSelector from './ModelSelector';

const Dashboard = () => {
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [stats, setStats] = useState({
    total_readings: 0,
    anomaly_count: 0,
    anomaly_percentage: 0,
    sensor_health: {}
  });
  const [isLoading, setIsLoading] = useState(true);

  // Fetch statistics
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/stats');
        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error('Error fetching stats:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStats();
    // Refresh stats every 30 seconds
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header Section */}
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">
          Real-Time Monitoring Dashboard
        </h2>
        <p className="text-gray-400">
          Intel Berkeley Research Lab - IoT Sensor Data Analysis
        </p>
      </div>

      {/* Model Selector */}
      <div className="mb-6">
        <ModelSelector
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
        />
      </div>

      {/* Statistics Panel */}
      <div className="mb-8">
        <StatisticsPanel stats={stats} isLoading={isLoading} />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Real-Time Monitor */}
        <div className="lg:col-span-2">
          <RealTimeMonitor selectedModel={selectedModel} />
        </div>

        {/* Anomaly Charts */}
        <div className="lg:col-span-2">
          <AnomalyChart />
        </div>
      </div>

      {/* Sensor Grid */}
      <div>
        <SensorGrid sensorHealth={stats.sensor_health} />
      </div>
    </div>
  );
};

export default Dashboard;
