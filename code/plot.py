
import matplotlib.pyplot as plt
import csv

def plot_results(episodes, rewards, avg_speeds, queue_lengths):
    # Plot Total Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(range(episodes), rewards, label="Total Rewards", color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward Across Episodes")
    plt.legend()
    plt.show()
    
    # Plot Average Speeds
    plt.figure(figsize=(12, 6))
    plt.plot(range(episodes), avg_speeds, label="Average Speed (m/s)", color='g')
    plt.xlabel("Episodes")
    plt.ylabel("Average Speed (m/s)")
    plt.title("Average Speed Across Episodes")
    plt.legend()
    plt.show()
    
    # Plot Queue Lengths
    plt.figure(figsize=(12, 6))
    plt.plot(range(episodes), queue_lengths, label="Average Queue Length", color='r')
    plt.xlabel("Episodes")
    plt.ylabel("Queue Length (vehicles)")
    plt.title("Queue Length Across Episodes")
    plt.legend()
    plt.show()