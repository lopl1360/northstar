-- Schema definition for the northstar storage layer.

CREATE TABLE IF NOT EXISTS symbols (
  symbol VARCHAR(24) PRIMARY KEY,
  exchange VARCHAR(8) NOT NULL,
  currency VARCHAR(8) NOT NULL,
  name VARCHAR(128),
  sector VARCHAR(64),
  is_active TINYINT(1) DEFAULT 1,
  first_seen DATE NOT NULL,
  last_seen DATE NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS daily_metrics (
  symbol VARCHAR(24) NOT NULL,
  ds DATE NOT NULL,
  price DECIMAL(18,6),
  vol BIGINT,
  pe_ttm DECIMAL(18,6),
  ps_ttm DECIMAL(18,6),
  ev_ebit DECIMAL(18,6),
  fcf_yield DECIMAL(18,6),
  roe DECIMAL(18,6),
  net_debt_ebitda DECIMAL(18,6),
  margin_trend_5y DECIMAL(18,6),
  sma_100 DECIMAL(18,6),
  sma_200 DECIMAL(18,6),
  atr_14 DECIMAL(18,6),
  earnings_within_48h TINYINT(1) DEFAULT 0,
  PRIMARY KEY (symbol, ds),
  INDEX idx_dm_ds (ds),
  INDEX idx_dm_comp (ds, price)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS valuations (
  symbol VARCHAR(24) NOT NULL,
  ds DATE NOT NULL,
  fv_dcf DECIMAL(18,6),
  dcf_upside DECIMAL(18,6),
  multiple_score DECIMAL(18,6),
  quality_score DECIMAL(18,6),
  momentum_score DECIMAL(18,6),
  composite_score DECIMAL(18,6),
  PRIMARY KEY (symbol, ds),
  INDEX idx_val_ds (ds),
  INDEX idx_val_comp (ds, composite_score)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS picks (
  ds DATE NOT NULL,
  rank_order TINYINT NOT NULL,
  symbol VARCHAR(24) NOT NULL,
  target_price DECIMAL(18,6) NOT NULL,
  stop_loss DECIMAL(18,6) NOT NULL,
  notes_json JSON,
  sent_telegram TINYINT(1) DEFAULT 0,
  PRIMARY KEY (ds, rank_order),
  INDEX idx_picks_symbol (symbol)
) ENGINE=InnoDB;
