import { useState, useEffect, useRef, useCallback } from 'react';
import { TelemetryData, SensorStatusMap } from '../types';

interface StreamingMessage {
  type: 'initial' | 'update' | 'pong' | 'incident_transition' | 'status';
  data?: TelemetryData[];
  transitions?: any;
  sensors?: SensorStatusMap;
}

interface StreamingOptions {
  websocketUrl: string;
  maxBufferSize?: number;
  reconnectInterval?: number;
  enableAutoReconnect?: boolean;
  onIncidentTransition?: (payload: any) => void;
}

export const useStreamingData = (options: StreamingOptions) => {
  const {
    websocketUrl,
    maxBufferSize = 1000,
    reconnectInterval = 3000,
    enableAutoReconnect = true,
    onIncidentTransition,
  } = options;

  const [data, setData] = useState<TelemetryData[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [lastUpdateTime, setLastUpdateTime] = useState<string>('');
  const [sensorStatus, setSensorStatus] = useState<SensorStatusMap>({});

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pingIntervalRef = useRef<NodeJS.Interval | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    console.log('🔌 Connecting to streaming data server...');
    setConnectionStatus('connecting');

    try {
      const ws = new WebSocket(websocketUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('connected');

        // Clear any pending reconnection
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }

        // Start ping interval to keep connection alive
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000); // Ping every 30 seconds
      };

      ws.onmessage = (event) => {
        try {
          const message: StreamingMessage = JSON.parse(event.data);

          if (message.sensors) {
            setSensorStatus(message.sensors);
          }

          if (message.type === 'initial' && message.data) {
            console.log('📊 Received initial data:', message.data.length, 'points');
            setData(message.data);
            setLastUpdateTime(new Date().toLocaleTimeString());
          } else if (message.type === 'update' && message.data) {
            // Add new data points and maintain buffer size
            setData(prevData => {
              const newData = [...prevData, ...message.data!];
              const trimmedData = newData.length > maxBufferSize
                ? newData.slice(-maxBufferSize)
                : newData;
              return trimmedData;
            });
            setLastUpdateTime(new Date().toLocaleTimeString());

            // Log streaming activity (throttled)
            if (message.data.length > 0) {
              console.log('📡 Streaming update:', message.data.length, 'new points');
            }
          } else if (message.type === 'incident_transition' && onIncidentTransition) {
            onIncidentTransition(message);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('🔌 WebSocket disconnected:', event.reason);
        setIsConnected(false);
        setConnectionStatus('disconnected');

        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        // Attempt to reconnect if enabled
        if (enableAutoReconnect) {
          console.log(`🔄 Reconnecting in ${reconnectInterval}ms...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setConnectionStatus('error');
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionStatus('error');
    }
  }, [websocketUrl, maxBufferSize, reconnectInterval, enableAutoReconnect, onIncidentTransition]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setConnectionStatus('disconnected');
  }, []);

  const reconnect = useCallback(() => {
    disconnect();
    setTimeout(connect, 100); // Small delay to ensure cleanup
  }, [disconnect, connect]);

  // Auto-connect on mount
  useEffect(() => {
    connect();

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Handle page visibility changes to manage connection
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Page is hidden, optionally disconnect to save resources
        // disconnect();
      } else {
        // Page is visible, ensure we're connected
        if (!isConnected && enableAutoReconnect) {
          connect();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isConnected, connect, enableAutoReconnect]);

  return {
    data,
    isConnected,
    connectionStatus,
    lastUpdateTime,
    sensorStatus,
    connect,
    disconnect,
    reconnect,
    // Utility functions
    getLatestData: () => data[data.length - 1],
    getDataCount: () => data.length,
    clearData: () => setData([])
  };
};
