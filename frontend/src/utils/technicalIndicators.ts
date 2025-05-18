export function calculateSMA(data: number[], period: number): (number | undefined)[] {
  if (period <= 0 || data.length < period) {
    return new Array(data.length).fill(undefined);
  }
  const sma: (number | undefined)[] = new Array(period - 1).fill(undefined);
  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += data[i];
  }
  sma.push(sum / period);
  for (let i = period; i < data.length; i++) {
    sum -= data[i - period];
    sum += data[i];
    sma.push(sum / period);
  }
  return sma;
}

export function calculateEMA(data: number[], period: number): (number | undefined)[] {
  if (period <= 0 || data.length < period) {
    return new Array(data.length).fill(undefined);
  }
  const ema: (number | undefined)[] = new Array(period - 1).fill(undefined);
  const multiplier = 2 / (period + 1);
  let sumInitial = 0;
  for (let i = 0; i < period; i++) {
    sumInitial += data[i];
  }
  let previousEma = sumInitial / period;
  ema.push(previousEma);

  for (let i = period; i < data.length; i++) {
    const currentEma = (data[i] - previousEma) * multiplier + previousEma;
    ema.push(currentEma);
    previousEma = currentEma;
  }
  return ema;
}

export function calculateRSI(data: number[], period: number = 14): (number | undefined)[] {
  if (period <= 0 || data.length < period + 1) {
    return new Array(data.length).fill(undefined);
  }

  const rsi: (number | undefined)[] = new Array(period).fill(undefined);
  let gains = 0;
  let losses = 0;

  for (let i = 1; i <= period; i++) {
    const diff = data[i] - data[i - 1];
    if (diff > 0) {
      gains += diff;
    } else {
      losses -= diff; // losses are positive
    }
  }

  let avgGain = gains / period;
  let avgLoss = losses / period;

  if (avgLoss === 0) {
    rsi.push(100);
  } else {
    const rs = avgGain / avgLoss;
    rsi.push(100 - (100 / (1 + rs)));
  }

  for (let i = period + 1; i < data.length; i++) {
    const diff = data[i] - data[i - 1];
    let currentGain = 0;
    let currentLoss = 0;

    if (diff > 0) {
      currentGain = diff;
    } else {
      currentLoss = -diff;
    }

    avgGain = (avgGain * (period - 1) + currentGain) / period;
    avgLoss = (avgLoss * (period - 1) + currentLoss) / period;

    if (avgLoss === 0) {
      rsi.push(100);
    } else {
      const rs = avgGain / avgLoss;
      rsi.push(100 - (100 / (1 + rs)));
    }
  }
  return rsi;
}

export interface MACDOutput {
  MACD?: number;
  signal?: number;
  histogram?: number;
}

export function calculateMACD(
  data: number[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): MACDOutput[] {
  if (data.length < slowPeriod || data.length < signalPeriod) {
    return new Array(data.length).fill({ MACD: undefined, signal: undefined, histogram: undefined });
  }

  const emaFast = calculateEMA(data, fastPeriod);
  const emaSlow = calculateEMA(data, slowPeriod);
  const macdLine: (number | undefined)[] = [];

  for (let i = 0; i < data.length; i++) {
    if (emaFast[i] !== undefined && emaSlow[i] !== undefined) {
      macdLine.push(emaFast[i]! - emaSlow[i]!);
    } else {
      macdLine.push(undefined);
    }
  }

  const signalLine = calculateEMA(macdLine.filter(val => val !== undefined) as number[], signalPeriod);
  const result: MACDOutput[] = [];

  let signalIndex = 0;
  for (let i = 0; i < data.length; i++) {
    const macdValue = macdLine[i];
    let signalValue: number | undefined = undefined;
    let histogramValue: number | undefined = undefined;

    if (macdValue !== undefined) {
      // Align signal line with MACD line data points
      // The signal line starts after `slowPeriod -1 + signalPeriod -1` original data points
      const macdDataPointsForSignal = macdLine.slice(slowPeriod - 1).filter(v => v !== undefined).length;
      if (i >= slowPeriod -1 && signalIndex < macdDataPointsForSignal && signalIndex < signalLine.length) {
         if(signalLine[signalIndex] !== undefined) {
            signalValue = signalLine[signalIndex];
            histogramValue = macdValue - signalValue!;
            signalIndex++;
         }
      }
    }
    result.push({ MACD: macdValue, signal: signalValue, histogram: histogramValue });
  }
  return result;
}

export interface BollingerBandsOutput {
  upper?: number;
  middle?: number;
  lower?: number;
}

export function calculateBollingerBands(
  data: number[],
  period: number = 20,
  stdDevMultiplier: number = 2
): BollingerBandsOutput[] {
  if (data.length < period) {
    return new Array(data.length).fill({ upper: undefined, middle: undefined, lower: undefined });
  }

  const sma = calculateSMA(data, period);
  const stdDevs: (number | undefined)[] = new Array(period - 1).fill(undefined);

  for (let i = period - 1; i < data.length; i++) {
    if (sma[i] === undefined) {
        stdDevs.push(undefined);
        continue;
    }
    let sumSqDiff = 0;
    for (let j = 0; j < period; j++) {
      sumSqDiff += Math.pow(data[i - j] - sma[i]!, 2);
    }
    stdDevs.push(Math.sqrt(sumSqDiff / period));
  }

  const result: BollingerBandsOutput[] = [];
  for (let i = 0; i < data.length; i++) {
    if (sma[i] !== undefined && stdDevs[i] !== undefined) {
      const middle = sma[i]!;
      const upper = middle + stdDevs[i]! * stdDevMultiplier;
      const lower = middle - stdDevs[i]! * stdDevMultiplier;
      result.push({ upper, middle, lower });
    } else {
      result.push({ upper: undefined, middle: undefined, lower: undefined });
    }
  }
  return result;
}

