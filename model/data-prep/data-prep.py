import time
import asyncio
import g4f
import json
import random

start = time.time()

async def process_api_request(request, index):
    while True:
        try:
            await asyncio.sleep(random.randint(10, 20))
            print(f"Started API request of index: {index}.")
            response = await g4f.ChatCompletion.create_async(
                model="llama2-70b",
                messages=[{"role": "user", "content": request}],
            )
            if len(response) == 0:
                continue
            print(f"Completed API request of index: {index}")
            return response

        except Exception as e:
            print(f"Request of index {index} - Error: {str(e)}")
            await asyncio.sleep(10)

async def run_concurrent_requests(concurrent_requests, manim_data):
    #async with Client() as session:
        tasks = []
        for index, (request, manim_sample) in enumerate(zip(concurrent_requests, manim_data)):
            tasks.append(process_api_request(request, index))
        return await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":

    with open('data_without_text_5k.json') as f:
        manim_data = json.load(f)

    random.shuffle(manim_data)
    start = time.time()
    requests = [f"I will give you a chunk of code that describes a neural network, can you describes it back to me like you are trying to ask me to illustrate it for you in casual tones. Get straight to the point and be precise about numerical values . This will be the code: \"{sample['code']}\"." for sample in manim_data]
    responses = asyncio.run(run_concurrent_requests(concurrent_requests=requests, manim_data=manim_data))
    end = time.time()
    print(f"Time elapsed: {end - start}")
    for resp, manim_sample in zip(responses, manim_data):
        manim_sample['text'] = resp

    json_object = json.dumps(manim_data, indent=4)

    with open("data.json", "w") as outfile:
        outfile.write(json_object)
