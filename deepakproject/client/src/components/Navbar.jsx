import React from 'react';

const Navbar = () => {
  return (
    <nav className="bg-gray-800 border-b border-gray-700 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <h1 className="text-2xl font-bold text-blue-400">
                IoT Anomaly Detection
              </h1>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <div className="h-3 w-3 bg-green-500 rounded-full mr-2 animate-pulse"></div>
              <span className="text-gray-300 text-sm">System Active</span>
            </div>
            <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors">
              Dashboard
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
