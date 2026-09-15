export const persona3aVerifiedRun = {
  status: 'completed' as const,
  completedAt: '2026-09-15T18:50:39.000Z',
  screenResponses: 780,
  screenScenarios: 12,
  selectedFeatures: 32,
  discoveryResponses: 1024,
  conditions: { baseline: 12, positive: 384, negative: 384 },
  truncatedResponses: 660,
  intervention: '± one calibrated amplitude (screenNormFraction = 0.1)',
  illustrativeSignals: [
    { feature: 51850, sign: '+', changedFraction: '80.8%', exactMatches: '0/12' },
    { feature: 38812, sign: '+', changedFraction: '77.0%', exactMatches: '0/12' },
    { feature: 32974, sign: '−', changedFraction: '49.4%', exactMatches: '2/12' },
    { feature: 101616, sign: '−', changedFraction: '54.3%', exactMatches: '1/12' },
  ],
  example: {
    scenario: 'Pug template backed by JSON or a database',
    baseline: 'JSON data with generic items rendered as a simple list.',
    positive: 'Feature 38812+: JSON data framed as a books list, with different wording and examples.',
  },
  featureIds: [
    101616, 51850, 38812, 27799, 8987, 48793, 41730, 106633,
    59977, 111998, 99519, 128631, 8136, 9232, 75898, 54450,
    127530, 126462, 51620, 116649, 104318, 52432, 65532, 99520,
    43431, 25027, 89177, 75055, 105108, 82843, 104155, 32974,
  ],
} as const;