-- Materialized views and indexes required by the API

-- Deck-level stats (side-agnostic)
CREATE MATERIALIZED VIEW IF NOT EXISTS deck_stats AS
WITH norm AS (
  SELECT a_deck_id AS deck_id, CASE WHEN a_won THEN 1 ELSE 0 END AS win
  FROM matches
  UNION ALL
  SELECT b_deck_id AS deck_id, CASE WHEN a_won THEN 0 ELSE 1 END AS win
  FROM matches
)
SELECT deck_id,
       COUNT(*)::INT    AS total_games,
       AVG(win::float)  AS winrate
FROM norm
GROUP BY deck_id;

CREATE UNIQUE INDEX IF NOT EXISTS deck_stats_uidx ON deck_stats(deck_id);

-- Named deck stats (includes card lists and names)
CREATE MATERIALIZED VIEW IF NOT EXISTS deck_stats_named AS
WITH names AS (
  SELECT d.id AS deck_id,
         d.card_ids,
         ARRAY(
           SELECT c.name
           FROM UNNEST(d.card_ids) AS cid
           JOIN cards c ON c.id = cid
           ORDER BY cid
         ) AS card_names
  FROM decks d
)
SELECT n.deck_id, n.card_ids, n.card_names, s.total_games, s.winrate
FROM names n
JOIN deck_stats s ON s.deck_id = n.deck_id;

CREATE UNIQUE INDEX IF NOT EXISTS deck_stats_named_uidx ON deck_stats_named(deck_id);

-- Directional matchup stats: P(A wins) for (a_deck_id, b_deck_id)
CREATE MATERIALIZED VIEW IF NOT EXISTS deck_matchups AS
SELECT a_deck_id, b_deck_id,
       COUNT(*)::INT AS games,
       AVG(CASE WHEN a_won THEN 1.0 ELSE 0.0 END) AS a_winrate
FROM matches
GROUP BY a_deck_id, b_deck_id;

CREATE UNIQUE INDEX IF NOT EXISTS deck_matchups_uidx ON deck_matchups(a_deck_id, b_deck_id);

-- Named matchup stats (includes the deck card arrays)
CREATE MATERIALIZED VIEW IF NOT EXISTS deck_matchups_named AS
SELECT m.a_deck_id,
       da.card_ids AS a_cards,
       m.b_deck_id,
       db.card_ids AS b_cards,
       m.games,
       m.a_winrate
FROM deck_matchups m
JOIN decks da ON da.id = m.a_deck_id
JOIN decks db ON db.id = m.b_deck_id;

CREATE UNIQUE INDEX IF NOT EXISTS deck_matchups_named_uidx 
ON deck_matchups_named(a_deck_id, b_deck_id);


