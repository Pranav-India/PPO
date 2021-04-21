import env
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.layers import Input , Conv2D, BatchNormalization, Dropout, Activation, Dense, Flatten ,MaxPooling2D

class PPO:
    def __init__(self):
        self._init_hyperparameters()
        self.env = env.Environment()
        self.obs_dim = self.env.observation_shape
        self.act_dim = self.env.action_space.shape[0]
        self.actor = self.create_model(self.act_dim)
        self.actor.compile(optimizer=Adam(learning_rate=self.lr))                                                  
        self.critic = self.create_model(1)
        self.critic.compile(optimizer=Adam(learning_rate=self.lr))
        
        self.con_mat = tf.eye(self.act_dim)*0.500

    def learn(self, total_timesteps):

        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            #print(batch_acts)
            t_so_far += np.sum(batch_lens)
            
            i_so_far += 1

            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V
            
            A_k = (A_k - tf.math.reduce_mean(A_k)) / (tf.math.reduce_std(A_k) + 1e-10)

            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
                with tf.GradientTape(persistent=True) as tape:    
					# Calculate V_phi and pi_theta(a_t | s_t)
                    V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                    ratios = tf.exp(curr_log_probs - tf.stop_gradient(batch_log_probs))

                    # Calculate surrogate losses.
                    A_k = tf.stop_gradient(A_k)
                    surr1 = ratios * A_k
                    surr2 = tf.clip_by_value(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    actor_loss = -tf.minimum(surr1, surr2)
                    actor_loss =  tf.reduce_mean(actor_loss)
                    mse = tf.keras.losses.MeanSquaredError()
                    critic_loss = mse(V, tf.stop_gradient(batch_rtgs))
                    # print(actor_loss ,critic_loss)
                    # Calculate gradients and perform backward propagation for actor network

                gradient_actor = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(gradient_actor, self.actor.trainable_variables))
                

                # Calculate gradients and perform backward propagation for critic network
                gradient_critic = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic.optimizer.apply_gradients(zip(gradient_critic, self.critic.trainable_variables))
                

    def rollout(self):
        # Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = []
        t = 0

        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation. 
            obs ,_ = self.env.reset()
            
            #print(type(obs))
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1 # Increment timesteps ran this batch so far
                batch_obs.append(obs)
                #print(obs , obs.shape)
                obs = np.asarray([obs])
                action, log_prob = self.get_action(obs)
                obs, rew, _ , done = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            print(sum(ep_rews))

        batch_obs = tf.convert_to_tensor(batch_obs, dtype=tf.float32)
        batch_acts = tf.convert_to_tensor(batch_acts, dtype=tf.float32)
        batch_log_probs = tf.convert_to_tensor(batch_log_probs, dtype=tf.float32)
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = tf.convert_to_tensor(batch_rtgs, dtype=tf.float32)

        return batch_rtgs

    def get_action(self, obs):
		
        # Query the actor network for a mean action
        #print(obs , obs.shape ,"get action" )
        mean = self.actor(obs)

        dist = tfp.distributions.MultivariateNormalDiag(mean, self.con_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.numpy(), log_prob

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs)
        V = tf.squeeze(V)

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = tfp.distributions.MultivariateNormalDiag(mean, self.con_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 300                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 50           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.0005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
    
    def create_model(self,act_dim):
        state_input = Input(self.obs_dim)

        conv_1 = Conv2D(32, (3, 3), activation='relu', input_shape=(75,250,3))(state_input)
        max_p_1 = MaxPooling2D((2, 2))(conv_1)
        conv_2 = Conv2D(64, (3, 3), activation='relu')(max_p_1)
        max_p_2 = MaxPooling2D((2, 2))(conv_2)
        conv_3 = Conv2D(64, (3, 3), activation='relu')(max_p_2)
        den_1 = Flatten()(conv_3)
        dense_1 = Dense(32, activation='relu')(den_1)
        dense_2 = Dense(32, activation='relu')(dense_1)
        out_mu = Dense(act_dim, activation='tanh')(dense_2)
        
        
        return tf.keras.models.Model(state_input, [out_mu])