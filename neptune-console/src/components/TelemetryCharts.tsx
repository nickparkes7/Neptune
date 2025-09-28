import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import { TelemetryData } from '../types';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

interface Props {
  data: TelemetryData[];
}

const TelemetryCharts: React.FC<Props> = ({ data }) => {
  // Prepare data for charts - only show last 100 points for performance
  const recentData = data.slice(-100);

  const timeLabels = recentData.map(d => new Date(d.ts));

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        type: 'time' as const,
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm',
          },
        },
      },
      y: {
        beginAtZero: true,
      },
    },
    elements: {
      point: {
        radius: 2,
      },
    },
    animation: {
      duration: 0, // Disable animations for real-time updates
    },
  };

  const oilData = {
    labels: timeLabels,
    datasets: [
      {
        data: recentData.map(d => ({ x: new Date(d.ts), y: d.oil_fluor_ppb })),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderWidth: 2,
        tension: 0.1,
      },
    ],
  };

  const chlorophyllData = {
    labels: timeLabels,
    datasets: [
      {
        data: recentData.map(d => ({ x: new Date(d.ts), y: d.chlorophyll_ug_per_l })),
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        borderWidth: 2,
        tension: 0.1,
      },
    ],
  };

  const backscatterData = {
    labels: timeLabels,
    datasets: [
      {
        data: recentData.map(d => ({ x: new Date(d.ts), y: d['backscatter_m-1_sr-1'] })),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        tension: 0.1,
      },
    ],
  };

  if (data.length === 0) {
    return (
      <div className="chart-container">
        <div className="alert alert-info">
          Waiting for live telemetry data...
        </div>
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ marginBottom: '1rem', color: '#e2e8f0' }}>
        Live Telemetry
      </h2>

      <div className="chart-container">
        <h3>Oil Fluorescence (ppb)</h3>
        <div style={{ height: '200px' }}>
          <Line data={oilData} options={chartOptions} />
        </div>
      </div>

      <div className="chart-container">
        <h3>Chlorophyll (μg/L)</h3>
        <div style={{ height: '200px' }}>
          <Line data={chlorophyllData} options={chartOptions} />
        </div>
      </div>

      <div className="chart-container">
        <h3>Backscatter (m⁻¹sr⁻¹)</h3>
        <div style={{ height: '200px' }}>
          <Line data={backscatterData} options={chartOptions} />
        </div>
      </div>
    </div>
  );
};

export default TelemetryCharts;