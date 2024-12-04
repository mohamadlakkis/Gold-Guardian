import base64
import requests

api_key = "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"


'''
1. **Trend Analysis:**
   - The price started at approximately 2740 USD/oz on November 3.
   - It decreased to around 2564 USD/oz by November 13.
   - The price then rose to about 2740 USD/oz by November 22.
   - It decreased again to approximately 2640 USD/oz by November 25, followed by a slight increase to 2659 USD/oz by November 29.

2. **Key Metrics:**
   - Highest Price: 2740.24 USD/oz.
   - Lowest Price: 2564.25 USD/oz.
   - Volatility: The price showed significant fluctuations, with a notable drop and subsequent rise within the month.

3. **Moving Averages:**
   - The chart does not display specific moving averages, but the price movements suggest potential short-term averages around the midpoints of the peaks and troughs.

4. **Historical Comparisons:**
   - The current data shows a pattern of sharp declines followed by recoveries, which may be compared to past volatility in the gold market.

5. **Market Indicators:**
   - Support Level: Around 2564 USD/oz, where the price rebounded.
   - Resistance Level: Approximately 2740 USD/oz, where the price peaked twice.

6. **Future Projections:**
   - If the current pattern continues, the price may stabilize around the 2650-2700 USD/oz range.
   - Potential for further volatility if external market conditions change.

7. **Investor Decision Support:**
   - Investors should be cautious of the volatility and consider short-term trading opportunities.
   - Monitoring for breakouts above resistance or drops below support could provide actionable signals.
   - Diversification and risk management are advised given the recent fluctuations.
'''


'''
https://docs.trychroma.com/guides
'''
def encode_image(image_path):
    """
    Encodes an image file to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_summary(image_data):
    """
    Extracts summary from an image using OpenAI Vision
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                                You are a financial assistant specializing in gold market analysis. Analyze the provided image/chart/graph and provide a comprehensive summary with a focus on key data points, trends, and actionable insights. Structure your analysis into clear, concise points covering the following:

                                1. Trend Analysis: Describe visible patterns in the data, such as upward, downward, or stable trends. Include specific numerical data (e.g., 'The price started at X, increased to Y, decreased to Z, and then rose to A') to illustrate the movement.

                                2. Key Metrics: Identify and report on significant metrics such as the highest and lowest prices, volatility, and notable price movements.

                                3. Moving Averages: Highlight any visible moving averages, specifying their values and any observed crossovers or deviations from the actual prices.

                                3. Historical Comparisons: Compare the current data to past trends, noting similarities, differences, or deviations from historical patterns.

                                4. Market Indicators: Identify significant indicators such as support and resistance levels, breakouts, or any other noteworthy market signals.

                                5. Future Projections: Provide insights on potential future movements or predictions, including the likelihood of trends continuing, reversing, or stabilizing.

                                6. Investor Decision Support: Summarize actionable insights to assist investors, including any warnings or opportunities related to the data.

                                Format the analysis as a list of points, ensuring clarity and relevance to investor decision-making and trend prediction.
                                Note: Your sole goal, is to provide the best summary of a chart and convert it into a text, so other models that understands only text can understand the image.
                                """
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    }
                ]
            }
        ],
        "max_tokens": 3000,
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    if "choices" in response_data and len(response_data["choices"]) > 0:
        return response_data["choices"][0]["message"]["content"]
    else:
        return "Could not extract text from image."


image_path = 'image.png'
image_data = encode_image(image_path)
description_image = extract_summary(image_data)
print(description_image)