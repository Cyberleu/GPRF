/*+ Leading((ci, rt)) */
create view v2 as
select
    ci.movie_id
from
    cast_info AS ci,
    role_type AS rt
where
    rt.id = ci.role_id;

/*+ Leading(((v1 mc)t)) */
explain
SELECT
    MIN(t.title) AS movie_with_american_producer
FROM
    v2,
    movie_companies AS mc,
    title AS t,
    char_name AS chn
WHERE
    t.production_year > 1974
    AND t.id = mc.movie_id
    AND t.id = v2.movie_id