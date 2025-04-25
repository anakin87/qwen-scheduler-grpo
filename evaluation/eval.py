import datasets
import argparse

import re

# lenient format check - only checks the schedule in order not to penalize non-reasoning models
overall_pattern = r".*<schedule>.*<event>.*<name>.*</name>.*<start>\d{2}:\d{2}</start>.*<end>\d{2}:\d{2}</end>.*</event>.*</schedule>.*"
overall_regex = re.compile(overall_pattern, re.DOTALL)

capture_pattern = r"""
    <event>\s*
        <name>([^<]+)</name>\s*
        <start>(\d{2}:\d{2})</start>\s*
        <end>(\d{2}:\d{2})</end>\s*
    </event>
"""

capture_regex = re.compile(capture_pattern, re.VERBOSE)


def get_events(content):
    """Extract event information from XML-like content.

    Args:
        content (str): XML-like string containing event data

    Returns:
        list: List of tuples (name, start_time, end_time)
    """
    return [
        (match.group(1), match.group(2), match.group(3))
        for match in capture_regex.finditer(content)
    ]


def time_to_minutes(time_str):
    """Convert a time string in HH:MM format to minutes.

    Args:
        time_str (str): Time string in HH:MM format

    Returns:
        int: Total number of minutes
    """
    hours, mins = map(int, time_str.split(":"))
    return hours * 60 + mins


def are_events_sorted(events):
    """Check if events are sorted by start time.

    Args:
        events (list): List of event tuples (name, start_time, end_time)

    Returns:
        bool: True if events are sorted by start time, False otherwise
    """
    return all(events[i][1] < events[i + 1][1] for i in range(len(events) - 1))


def find_overlaps(events):
    """Detect if any events in the schedule overlap in time.

    Args:
        events (list): List of event tuples (name, start_time, end_time)

    Returns:
        bool: True if any events overlap, False otherwise
    """
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            start1, end1 = events[i][1], events[i][2]
            start2, end2 = events[j][1], events[j][2]
            if (start1 < end2) and (end1 > start2):
                return True
    return False


def compute_score(events, original_priorities):
    """Calculate the score for a schedule based on event durations and priorities.

    Args:
        events (list): List of event tuples (name, start_time, end_time)
        original_priorities (list): List of event names that have priority

    Returns:
        int: Total score for the schedule
    """
    score = 0
    for event in events:
        weigth = 2 if event[0] in original_priorities else 1
        score += weigth * (event[2] - event[1])
    return score


def evaluate(results_path):
    """Evaluate a set of scheduling results against various criteria.

    This function processes a directory of result files, checking each schedule for:
    - Format compliance
    - Minimum event count
    - Event existence
    - Temporal ordering
    - Overlaps
    - Priority-based scoring

    Args:
        results_path (str): Path to the directory containing result files

    Returns:
        None: Prints evaluation statistics to stdout
    """

    ds = datasets.load_dataset("anakin87/events-scheduling")
    scores = []

    (
        format_errors,
        less_than_2_events,
        overlaps,
        unsorted,
        non_existing,
        valid_schedules,
    ) = 0, 0, 0, 0, 0, 0

    for i in range(0, 100):
        with open(f"{results_path}/{i}.txt", "r") as f:
            content = f.read()

        original_example = ds["test"][i]

        # overall format check
        if not overall_regex.match(content):
            scores.append(0)
            format_errors += 1
            continue

        # parse XML
        events = get_events(content)

        if len(events) < 2:
            scores.append(0)
            less_than_2_events += 1
            continue

        existing_events = {
            ev for ev in events if [ev[0], ev[1], ev[2]] in original_example["events"]
        }
        if len(existing_events) < len(events):
            scores.append(0)
            non_existing += 1
            continue

        events_minutes = [
            (ev[0], time_to_minutes(ev[1]), time_to_minutes(ev[2])) for ev in events
        ]

        # check for sorted events
        if not are_events_sorted(events_minutes):
            scores.append(0)
            unsorted += 1
            continue

        # check for overlaps
        if find_overlaps(events_minutes):
            scores.append(0)
            overlaps += 1
            continue

        priority_events = original_example["priority_events"]

        # compute score
        score = compute_score(events_minutes, priority_events)
        optimal_score = original_example["optimal_score"]
        scores.append(score / optimal_score)
        valid_schedules += 1

    print(f"Format errors: {format_errors}")
    print(f"Less than 2 events: {less_than_2_events}")
    print(f"Overlaps: {overlaps}")
    print(f"Unsorted: {unsorted}")
    print(f"Non existing: {non_existing}")
    print(f"Valid schedules: {valid_schedules}")
    print(f"Average score: {sum(scores) / len(scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate results with configurable path"
    )

    parser.add_argument(
        "--path", type=str, help="Path to the results directory to evaluate"
    )

    args = parser.parse_args()

    evaluate(args.path)
