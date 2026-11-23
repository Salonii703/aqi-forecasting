import math


def aqi_category(aqi):
    """Return the Indian AQI category and corresponding health advice."""

    if not isinstance(aqi, (int, float)) or not math.isfinite(aqi):
        return "Invalid AQI", "Please provide a valid AQI value."

    if aqi < 0:
        return "Invalid AQI", "AQI cannot be negative."

    if aqi <= 50:
        return (
            "Good",
            "Air quality is satisfactory. Enjoy outdoor activities."
        )

    elif aqi <= 100:
        return (
            "Satisfactory",
            "Sensitive people should limit prolonged outdoor exertion."
        )

    elif aqi <= 200:
        return (
            "Moderate",
            "People with asthma or heart disease should reduce outdoor activity."
        )

    elif aqi <= 300:
        return (
            "Poor",
            "Avoid prolonged outdoor exertion; consider wearing a well-fitting mask outdoors."
        )

    elif aqi <= 400:
        return (
            "Very Poor",
            "Avoid outdoor activity where possible. Consider using an air purifier indoors."
        )

    else:
        return (
            "Severe",
            "Minimize outdoor exposure. Vulnerable people should take extra precautions and seek medical advice if symptoms occur."
        )


# --------------------------------------------------
# Test the health alert logic
# --------------------------------------------------

test_aqi_values = [35, 85, 150, 250, 350, 450]

print("AQI HEALTH ALERTS")
print("=" * 60)

for value in test_aqi_values:
    category, advice = aqi_category(value)

    print(f"\nPredicted AQI: {value}")
    print(f"Category: {category}")
    print(f"Health Advice: {advice}")
    print("-" * 60)