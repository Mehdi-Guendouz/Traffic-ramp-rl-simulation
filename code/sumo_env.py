import traci
import numpy as np



# Define SUMO environment
class SumoRampEnv:
    def __init__(self):
        self.sumoCmd = [
            "sumo",
             "-c", r"D:\3CS\RL\Reinforcement-learning-project-for-ramp-metering-on-highways-main\RL_project.sumocfg",
        ]
        self.actions = [0, 1, 2]  # Green, Yellow, Red
        self.state_size = 6  # State dimensions
        self.action_size = len(self.actions)
        self.reset()

    def reset(self):
        try:
            traci.close()
        except Exception:
            pass
        # print("Starting SUMO with command:", " ".join(self.sumoCmd))
        traci.start(self.sumoCmd)
        try:
            traci.start(self.sumoCmd)
            print("SUMO started successfully.")
        except Exception as e:
            print(f"Error starting SUMO: {e}")
        return self.get_state()
    
    def step(self, action):
        try:
            if action == 0:
                traci.trafficlight.setPhase("ramp_metering_tl", 0)  # Green
            elif action == 1:
                traci.trafficlight.setPhase("ramp_metering_tl", 1)  # Yellow
            else:
                traci.trafficlight.setPhase("ramp_metering_tl", 2)  # Red

            traci.simulationStep()
            next_state = self.get_state()
            reward = self.calculate_reward()
            done = traci.simulation.getMinExpectedNumber() <= 0
            return next_state, reward, done
        except traci.exceptions.TraCIException as e:
            print(f"Error in step function: {e}")
            return np.zeros(self.state_size), -1, True
    
    def get_state(self):
        try:
            # Raw features from SUMO
            highway_density = traci.edge.getLastStepVehicleNumber("highway_entry") / 3   
            ramp_density = traci.edge.getLastStepVehicleNumber("ramp_entry")
            avg_speed = traci.edge.getLastStepMeanSpeed("highway_entry")
            queue_length = traci.edge.getLastStepHaltingNumber("ramp_entry")
            traffic_light_phase = traci.trafficlight.getPhase("ramp_metering_tl")
            phase_duration = traci.trafficlight.getPhaseDuration("ramp_metering_tl")

            # Normalization: Define max values for each feature
            max_highway_density = 50  # Example: Adjust based on simulation scale
            max_ramp_density = 20
            max_avg_speed = 33.33  # Example: 120 km/h in m/s
            max_queue_length = 30
            max_phase_duration = 60  # Example: Assume max phase is 60 seconds

            # Normalize features (scaling to 0-1 range)
            highway_density_norm = min(highway_density / max_highway_density, 1.0)
            ramp_density_norm = min(ramp_density / max_ramp_density, 1.0)
            avg_speed_norm = avg_speed / max_avg_speed
            queue_length_norm = min(queue_length / max_queue_length, 1.0)
            phase_duration_norm = phase_duration / max_phase_duration

            # Traffic light phase is categorical; one-hot encoding could be used instead.
            # For simplicity, we scale to 0-1.
            traffic_light_phase_norm = traffic_light_phase / 2.0  # Assuming 3 phases (0, 1, 2)

            # Construct normalized state
            return np.array([
                highway_density_norm, ramp_density_norm, avg_speed_norm, 
                queue_length_norm, traffic_light_phase_norm, phase_duration_norm
            ])
        except traci.exceptions.TraCIException as e:
            print(f"Error during state retrieval: {e}")
            return np.zeros(self.state_size)


    def calculate_reward(self):
        throughput = traci.edge.getLastStepVehicleNumber("highway_entry")
        ramp_queue = traci.edge.getLastStepHaltingNumber("ramp_entry")
        avg_speed = traci.edge.getLastStepMeanSpeed("highway_entry")

        alpha = 0.7
        beta = -4
        gamma = 1.5

        reward = (alpha * throughput) + (beta * ramp_queue) + (gamma * avg_speed)

        if avg_speed < 2.0:
            reward -= 2.0  # Penalty for severe congestion

        collisions = traci.simulation.getCollidingVehiclesNumber()
        if collisions > 0:
            reward -= 5.0  # Moderate collision penalty

        return reward

    def close(self):
        traci.close()