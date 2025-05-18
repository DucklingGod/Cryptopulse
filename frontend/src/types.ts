export interface USDQuote {
  price: number;
  volume_24h: number;
  volume_change_24h: number;
  percent_change_1h: number;
  percent_change_24h: number;
  percent_change_7d: number;
  percent_change_30d: number;
  percent_change_60d: number;
  percent_change_90d: number;
  market_cap: number;
  market_cap_dominance: number;
  fully_diluted_market_cap: number;
  tvl: number | null;
  last_updated: string;
}

export interface Coin {
  id: number;
  name: string;
  symbol: string;
  slug: string;
  num_market_pairs: number;
  date_added: string;
  tags: string[];
  max_supply: number | null;
  circulating_supply: number;
  total_supply: number;
  infinite_supply: boolean;
  platform: null | { id: number; name: string; slug: string; symbol: string; token_address: string; };
  cmc_rank: number;
  self_reported_circulating_supply: number | null;
  self_reported_market_cap: number | null;
  tvl_ratio: number | null;
  last_updated: string;
  quote: {
    USD: USDQuote;
  };
}

export interface NewsSource {
  title: string;
  region: string;
  domain: string;
  path: string | null;
}

export interface NewsCurrency {
  code: string;
  title: string;
  slug: string;
  url: string;
}

export interface NewsVotes {
  negative: number;
  positive: number;
  important: number;
  liked: number;
  disliked: number;
  lol: number;
  toxic: number;
  saved: number;
  comments: number;
}

export interface NewsItem {
  kind: string;
  domain: string;
  source: NewsSource;
  title: string;
  published_at: string;
  slug: string;
  id: number;
  url: string;
  created_at: string;
  currencies?: NewsCurrency[];
  votes: NewsVotes;
}

export interface HistoricalMeta {
  symbol: string;
  interval: string;
  currency_base: string;
  currency_quote: string;
  exchange: string;
  type: string;
}

export interface HistoricalValue {
  datetime: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume?: string; // Optional as it wasn't in the example but often present
}

export interface HistoricalData {
  meta?: HistoricalMeta;
  values?: HistoricalValue[];
  status?: string;
  message?: string; // For error messages from API
}

// Risk Assessment Types
export interface RiskComponent {
  score: number;
  weight: number;
}

export interface FearGreedComponent extends RiskComponent {
  value: number;
  classification: string;
}

export interface RiskAssessment {
  symbol: string;
  timestamp: string;
  overall_risk: {
    score: number;
    level: string;
  };
  components: {
    volatility: RiskComponent;
    sentiment: RiskComponent;
    prediction: RiskComponent;
    fear_greed_index: FearGreedComponent;
  };
}

export interface FearGreedIndex {
  value: number;
  classification: string;
  timestamp: string;
}

export interface MLPrediction {
  predicted_next_close: number;
  trend: string;
  last_actual_close: number;
  mock_implementation?: boolean;
  confidence?: number;
}

export interface MarketTrend {
  shortTerm: string;
  mediumTerm: string;
  longTerm: string;
  overall: string;
}
