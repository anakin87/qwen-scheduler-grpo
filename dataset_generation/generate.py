"""
Event Scheduling Dataset Generator

This module generates synthetic event scheduling problems with the following characteristics:
- Random number of events (4-8) with varying durations
- Events can overlap with a controlled probability
- Some events are marked as priority events
- Each problem includes an optimal score calculation
"""

import random
from bisect import bisect_right
import json
import datasets

# Set seeds for reproducibility
random.seed(42)

# Event generation constants
MAX_EVENTS = 8
MIN_EVENTS = 4
DURATIONS = [15, 30, 45, 60, 75, 90, 105, 120]
MAX_START_HOUR = 21  # Ensures events finish within the day

# Overlap probability and constraints
OVERLAP_PROBABILITY = 0.2
MIN_OVERLAPS = 1  # Must have at least one overlap
MAX_OVERLAP_RATIO = 0.4  # Maximum 40% of events can overlap

# Priority selection constraints
MIN_PRIORITY_RATIO = 0.2  # Minimum 20% of events are priority
MAX_PRIORITY_RATIO = 0.4  # Maximum 40% of events are priority

# Load the events categories names
with open("events_categories_names.json", "r") as f:
    events_categories_names = json.load(f)

def minutes_to_time(minutes):
    """Convert minutes since midnight to HH:MM time string.
    
    Args:
        minutes (int): Number of minutes since midnight
        
    Returns:
        str: Time in HH:MM format
    """
    return f"{minutes // 60:02d}:{minutes % 60:02d}"

def time_to_minutes(time_str):
    """Convert HH:MM time string to minutes since midnight.
    
    Args:
        time_str (str): Time in HH:MM format
        
    Returns:
        int: Number of minutes since midnight
    """
    hours, mins = map(int, time_str.split(':'))
    return hours * 60 + mins

def count_overlapping_events(events):
    """Count the number of overlapping event pairs in a schedule.
    
    Args:
        events (list): List of events, where each event is a tuple of (name, start_time, end_time)
        
    Returns:
        int: Number of overlapping event pairs
    """
    overlapping_count = 0
    for j in range(len(events)):
        for k in range(j + 1, len(events)):
            e1_start = time_to_minutes(events[j][1])
            e1_end = time_to_minutes(events[j][2])
            e2_start = time_to_minutes(events[k][1])
            e2_end = time_to_minutes(events[k][2])
            
            if (e1_start <= e2_end and e2_start <= e1_end):
                overlapping_count += 1
    return overlapping_count

def random_event():
    """Generate a random event with random start time and duration.
    
    Returns:
        tuple: (start_time, end_time) in HH:MM format
    """
    start_mins = random.randint(0, MAX_START_HOUR * 60 + 59)
    duration = random.choice(DURATIONS)
    end_mins = start_mins + duration
    return minutes_to_time(start_mins), minutes_to_time(end_mins)

def overlapping_event(prev_event):
    """Generate an event that overlaps with a previous event.
    
    Args:
        prev_event (tuple): Previous event tuple (name, start_time, end_time)
        
    Returns:
        tuple: (start_time, end_time) in HH:MM format
    """
    prev_start_mins = time_to_minutes(prev_event[1])
    prev_end_mins = time_to_minutes(prev_event[2])
    start_mins = random.randint(prev_start_mins, prev_end_mins - 1)
    duration = random.choice(DURATIONS)
    end_mins = start_mins + duration
    return minutes_to_time(start_mins), minutes_to_time(end_mins)

def generate_events():
    """Generate a valid schedule of events with controlled overlap and priority constraints.
    
    Returns:
        tuple: (events, priority_list) where:
            - events: List of (name, start_time, end_time) tuples
            - priority_list: List of event names marked as priority
    """
    category = random.choice(list(events_categories_names.keys()))    
    while True:  # Keep trying until we get a valid schedule
        event_names = list(events_categories_names[category])  # create a copy

        events = []
        n_events = random.randint(MIN_EVENTS, MAX_EVENTS)

        for i in range(1, n_events + 1):
            
            event_name = random.choice(event_names)
            event_names.remove(event_name)
            
            if i == 1 or random.random() >= OVERLAP_PROBABILITY or not events:
                event_start, event_end = random_event()
            else:
                event_start, event_end = overlapping_event(random.choice(events))
            
            events.append((event_name, event_start, event_end))
            events.sort(key=lambda x: x[1])

        total_overlaps = count_overlapping_events(events)
        
        # Check if we have a valid schedule
        if total_overlaps >= MIN_OVERLAPS and total_overlaps <= MAX_OVERLAP_RATIO * len(events):
            # Select priority events
            min_priority = max(1, int(len(events) * MIN_PRIORITY_RATIO))
            max_priority = int(len(events) * MAX_PRIORITY_RATIO)
            n_priority = random.randint(min_priority, max_priority)
            
            priority_events = random.sample(events, n_priority)
            priority_list = [e[0] for e in priority_events]

            events = sorted(events, key=lambda x: time_to_minutes(x[1]))
            
            return events, priority_list

def compute_optimal_score(events, priority_list):
    """Compute the optimal score for a schedule using dynamic programming.
    
    This implements a weighted interval scheduling algorithm where priority events
    have double the weight of regular events.
    We want to maximize the total weighted duration of the events.

    Inspired by: https://algo.monster/problems/1235
    
    Args:
        events (list): List of (name, start_time, end_time) tuples
        priority_list (list): List of event names marked as priority
        
    Returns:
        int: Maximum possible score for the schedule
    """
    start_times = []
    end_times = []
    profits = []
    for event in events:
        start_times.append(time_to_minutes(event[1]))
        end_times.append(time_to_minutes(event[2]))
        weight = 2 if event[0] in priority_list else 1
        duration = time_to_minutes(event[2]) - time_to_minutes(event[1])
        profits.append(weight*duration)

    
    # Combine the job information into a single list and sort by end time.
    jobs = sorted(zip(end_times, start_times, profits))
    
    # Get the total number of jobs.
    number_of_jobs = len(jobs)
    
    # Initialize dynamic programming table with 0 profits.
    dp = [0] * (number_of_jobs + 1)
    
    # Iterate over the jobs.
    for i, (current_end_time, current_start_time, current_profit) in enumerate(jobs):
        # Find the rightmost job that doesn't conflict with the current job's start time.
        # Use binary search for efficient querying. 'hi' is set to the current index 'i' for optimization.
        index = bisect_right(jobs, current_start_time, hi=i, key=lambda x: x[0])
        
        # Update the DP table by choosing the maximum of either taking the current job or not.
        # If taking the current job, add its profit to the total profit of non-conflicting jobs.
        dp[i + 1] = max(dp[i], dp[index] + current_profit)
    
    # Return the maximum profit which is the last element in the DP table.
    return dp[number_of_jobs]

def generate_row():
    """Generate a single scheduling problem with all required information.
    
    Returns:
        dict: Dictionary containing:
            - events: List of events with times
            - priority_events: List of priority event names
            - optimal_score: Maximum possible score
            - prompt: Human-readable description of the problem
    """
    dict_events = {}
    events, priority_list = generate_events()
    dict_events["events"] = events
    dict_events["priority_events"] = priority_list

    dict_events["optimal_score"] = compute_optimal_score(events, priority_list)

    prompt = "Events:\n"+"\n".join([f"- {event[0]} ({event[1]} - {event[2]})" for event in events])+ "\n\n"
    prompt += "Priorities:\n" + "\n".join([f"- {priority}" for priority in priority_list])
    dict_events["prompt"] = prompt

    return dict_events

def generate_dataset():
    """Generate a complete dataset of scheduling problems and upload to Hugging Face."""
    dataset_list = []
    for _ in range(600):
        dataset_list.append(generate_row())
    dataset = datasets.Dataset.from_list(dataset_list)
    dataset = dataset.train_test_split(test_size=100, seed=42)
    # uncomment to push the dataset
    # dataset.push_to_hub("anakin87/events-scheduling")

if __name__ == "__main__":
    generate_dataset()
