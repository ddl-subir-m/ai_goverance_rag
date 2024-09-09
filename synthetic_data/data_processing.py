
def process_data(data):
    # This function processes incoming data
    cleaned_data = [item.strip().lower() for item in data if item]
    return cleaned_data

def analyze_results(results):
    # This function analyzes the results of data processing
    total = sum(results)
    average = total / len(results) if results else 0
    return {"total": total, "average": average}
