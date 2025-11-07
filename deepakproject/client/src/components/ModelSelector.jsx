import React from 'react';

const ModelSelector = ({ selectedModel, setSelectedModel }) => {
  const models = [
    {
      id: 'ensemble',
      name: 'Ensemble',
      description: 'Combined predictions from multiple models',
      icon: '🎯',
      color: 'purple',
    },
    {
      id: 'isolation_forest',
      name: 'Isolation Forest',
      description: 'Unsupervised tree-based anomaly detection',
      icon: '🌲',
      color: 'green',
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      description: 'Supervised ensemble classification',
      icon: '🌳',
      color: 'blue',
    },
    {
      id: 'xgboost',
      name: 'XGBoost',
      description: 'Gradient boosting with high performance',
      icon: '⚡',
      color: 'yellow',
    },
    {
      id: 'autoencoder',
      name: 'Autoencoder',
      description: 'Deep learning reconstruction-based detection',
      icon: '🧠',
      color: 'pink',
    },
  ];

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg border border-gray-700">
      <h3 className="text-xl font-bold text-white mb-4">Select Detection Model</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        {models.map((model) => (
          <button
            key={model.id}
            onClick={() => setSelectedModel(model.id)}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedModel === model.id
                ? `border-${model.color}-500 bg-${model.color}-500/20`
                : 'border-gray-600 bg-gray-700 hover:border-gray-500'
            }`}
          >
            <div className="text-center">
              <div className="text-3xl mb-2">{model.icon}</div>
              <div className="font-bold text-white mb-1">{model.name}</div>
              <div className="text-xs text-gray-400">{model.description}</div>
            </div>
          </button>
        ))}
      </div>

      {/* Selected Model Info */}
      <div className="mt-4 p-4 bg-gray-700 rounded-lg border border-gray-600">
        <div className="flex items-center">
          <span className="text-gray-400 text-sm">Currently Selected:</span>
          <span className="ml-2 font-semibold text-white">
            {models.find((m) => m.id === selectedModel)?.name}
          </span>
        </div>
      </div>
    </div>
  );
};

export default ModelSelector;
