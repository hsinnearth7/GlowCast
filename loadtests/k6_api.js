/**
 * GlowCast API Load Test — k6 script
 *
 * Tests all major API endpoints under realistic load patterns:
 * - Health check (high frequency, low latency SLO)
 * - Forecast retrieval (primary read path)
 * - Pipeline trigger (write path, lower frequency)
 * - Drift status (monitoring path)
 *
 * Usage:
 *   k6 run loadtests/k6_api.js
 *   k6 run --vus 50 --duration 5m loadtests/k6_api.js
 *   k6 run --env BASE_URL=http://production-url loadtests/k6_api.js
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'changeme-dev';

const HEADERS = {
  'Content-Type': 'application/json',
  'X-API-Key': API_KEY,
};

// Custom metrics
const errorRate = new Rate('glowcast_errors');
const forecastLatency = new Trend('glowcast_forecast_latency', true);
const healthLatency = new Trend('glowcast_health_latency', true);
const pipelineLatency = new Trend('glowcast_pipeline_latency', true);
const driftLatency = new Trend('glowcast_drift_latency', true);
const requestsTotal = new Counter('glowcast_requests_total');

// ---------------------------------------------------------------------------
// Test scenarios
// ---------------------------------------------------------------------------

export const options = {
  scenarios: {
    // Sustained load — simulates normal traffic
    sustained: {
      executor: 'constant-vus',
      vus: 10,
      duration: '3m',
      startTime: '0s',
    },

    // Ramp up — simulates traffic increase
    ramp_up: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 20 },
        { duration: '2m', target: 50 },
        { duration: '1m', target: 50 },
        { duration: '1m', target: 0 },
      ],
      startTime: '3m',
    },

    // Spike — simulates sudden traffic burst
    spike: {
      executor: 'ramping-vus',
      startVUs: 5,
      stages: [
        { duration: '10s', target: 100 },
        { duration: '30s', target: 100 },
        { duration: '10s', target: 5 },
      ],
      startTime: '8m',
    },
  },

  // SLO thresholds
  thresholds: {
    // Availability: 99.9% of requests succeed
    'glowcast_errors': [{ threshold: 'rate<0.001', abortOnFail: false }],

    // Latency P95 < 500ms for all endpoints
    'http_req_duration': [
      { threshold: 'p(95)<500', abortOnFail: false },
      { threshold: 'p(99)<1000', abortOnFail: false },
    ],

    // Health check: P95 < 100ms
    'glowcast_health_latency': [{ threshold: 'p(95)<100', abortOnFail: false }],

    // Forecast: P95 < 500ms
    'glowcast_forecast_latency': [{ threshold: 'p(95)<500', abortOnFail: false }],

    // Pipeline trigger: P95 < 2000ms (async, returns immediately)
    'glowcast_pipeline_latency': [{ threshold: 'p(95)<2000', abortOnFail: false }],

    // Drift status: P95 < 500ms
    'glowcast_drift_latency': [{ threshold: 'p(95)<500', abortOnFail: false }],
  },
};

// ---------------------------------------------------------------------------
// Test functions
// ---------------------------------------------------------------------------

// Generate a random SKU ID (SKU_0001 to SKU_5000)
function randomSKU() {
  const id = Math.floor(Math.random() * 5000) + 1;
  return `SKU_${String(id).padStart(4, '0')}`;
}

export default function () {
  // Weight distribution: 40% health, 40% forecast, 10% drift, 10% pipeline
  const roll = Math.random();

  if (roll < 0.40) {
    testHealthCheck();
  } else if (roll < 0.80) {
    testForecast();
  } else if (roll < 0.90) {
    testDriftStatus();
  } else {
    testPipelineTrigger();
  }

  sleep(Math.random() * 2 + 0.5); // 0.5-2.5s think time
}

function testHealthCheck() {
  group('Health Check', function () {
    const res = http.get(`${BASE_URL}/api/health`);

    requestsTotal.add(1);
    healthLatency.add(res.timings.duration);

    const passed = check(res, {
      'health: status 200': (r) => r.status === 200,
      'health: status ok': (r) => {
        try {
          return JSON.parse(r.body).status === 'ok';
        } catch {
          return false;
        }
      },
      'health: latency < 100ms': (r) => r.timings.duration < 100,
    });

    if (!passed) {
      errorRate.add(1);
    } else {
      errorRate.add(0);
    }
  });
}

function testForecast() {
  group('Forecast', function () {
    const sku = randomSKU();
    const horizon = Math.floor(Math.random() * 28) + 7; // 7-35 days

    const res = http.get(
      `${BASE_URL}/api/forecasts/${sku}?horizon=${horizon}`,
      { headers: HEADERS }
    );

    requestsTotal.add(1);
    forecastLatency.add(res.timings.duration);

    const passed = check(res, {
      'forecast: status 200': (r) => r.status === 200,
      'forecast: has sku_id': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.sku_id === sku;
        } catch {
          return false;
        }
      },
      'forecast: has forecasts array': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body.forecasts) && body.forecasts.length > 0;
        } catch {
          return false;
        }
      },
      'forecast: has confidence intervals': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body.confidence_intervals);
        } catch {
          return false;
        }
      },
      'forecast: correct horizon': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.horizon_days === horizon;
        } catch {
          return false;
        }
      },
      'forecast: latency < 500ms': (r) => r.timings.duration < 500,
    });

    if (!passed) {
      errorRate.add(1);
    } else {
      errorRate.add(0);
    }
  });
}

function testDriftStatus() {
  group('Drift Status', function () {
    const res = http.get(`${BASE_URL}/api/drift/status`, {
      headers: HEADERS,
    });

    requestsTotal.add(1);
    driftLatency.add(res.timings.duration);

    const passed = check(res, {
      'drift: status 200': (r) => r.status === 200,
      'drift: has overall_status': (r) => {
        try {
          const body = JSON.parse(r.body);
          return ['healthy', 'warning', 'critical'].includes(body.overall_status);
        } catch {
          return false;
        }
      },
      'drift: has checks array': (r) => {
        try {
          return Array.isArray(JSON.parse(r.body).checks);
        } catch {
          return false;
        }
      },
      'drift: latency < 500ms': (r) => r.timings.duration < 500,
    });

    if (!passed) {
      errorRate.add(1);
    } else {
      errorRate.add(0);
    }
  });
}

function testPipelineTrigger() {
  group('Pipeline Trigger', function () {
    const payload = JSON.stringify({
      pipeline: 'data_generation',
      n_skus: 50,
      n_days: 90,
    });

    const res = http.post(`${BASE_URL}/api/pipelines/run`, payload, {
      headers: HEADERS,
    });

    requestsTotal.add(1);
    pipelineLatency.add(res.timings.duration);

    const passed = check(res, {
      'pipeline: status 200': (r) => r.status === 200,
      'pipeline: has pipeline_id': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.pipeline_id && body.pipeline_id.startsWith('pipeline-');
        } catch {
          return false;
        }
      },
      'pipeline: status pending or running': (r) => {
        try {
          const body = JSON.parse(r.body);
          return ['pending', 'running'].includes(body.status);
        } catch {
          return false;
        }
      },
      'pipeline: latency < 2000ms': (r) => r.timings.duration < 2000,
    });

    if (!passed) {
      errorRate.add(1);
    } else {
      errorRate.add(0);
    }

    // Poll for status
    if (res.status === 200) {
      try {
        const body = JSON.parse(res.body);
        const statusRes = http.get(
          `${BASE_URL}/api/pipelines/status?pipeline_id=${body.pipeline_id}`,
          { headers: HEADERS }
        );
        check(statusRes, {
          'pipeline status: accessible': (r) => r.status === 200,
        });
      } catch {
        // ignore parse errors
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Lifecycle hooks
// ---------------------------------------------------------------------------

export function setup() {
  // Verify the API is reachable before running tests
  const res = http.get(`${BASE_URL}/api/health`);
  if (res.status !== 200) {
    throw new Error(`API not reachable at ${BASE_URL}: status ${res.status}`);
  }
  console.log(`API verified at ${BASE_URL} — starting load test`);
  return { baseUrl: BASE_URL };
}

export function teardown(data) {
  console.log(`Load test complete against ${data.baseUrl}`);
}
