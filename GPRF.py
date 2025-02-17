
class GPRF:
   def  __init__(self, agent, args, experiences):
    
   def run(self):
       env = self.env_plan_heap(self.env_config)
        exhausted = True
        while True:
            print("run")
            if (exhausted or len(env.complete_plans) == self.num_complete_plans):
                if self.episode.value >= self.total_episodes * self.n_queries:
                    return
                with self.episode.get_lock():
                    query_num = self.episode.value % self.n_queries
                    if query_num == 0:
                        if self.sync:
                            self.step_flag.clear()
                        np.random.shuffle(self.random_query_ids)
                    randomized_query_idx = self.random_query_ids[query_num]
                    env.reset(randomized_query_idx)
                    self.episode.value += 1
                self.step_flag.wait()
            sup_plans = env.valid_actions()
            _, _, _, exhausted = env.step(*self.agent.predict(sup_plans))
            with self.step.get_lock():
                self.step.value += 1
            if (exhausted or len(env.complete_plans) == self.num_complete_plans):
                LOG.debug(
                    f"Completed plans for {env.query_id} query with costs = {env.costs}")
                self.update_q.put(
                    (env.complete_plans, env.costs, len(env.min_heap), env.query_id)) 