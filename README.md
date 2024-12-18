# fake_product_review_detection




### 💻 Use it in Your Python Code
```python
import requests

url = ""
payload = {
    "review": # Write your review
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    prediction = data['prediction']
    confidence = data['confidence']
    print("The review is :",prediction," with Confidence : ", confidence)
    
else:
    print("Error:", response.status_code, response.text)
```


### 💻 Use it in Your JavaScript Code
```javascript
const url = "";

const payload = {
    "review": # Write your review
};

fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => {
    console.log("prediction:", data.prediction);
    console.log("confidence:", data.confidence);
  })
  .catch(error => {
    console.error("Error:", error);
  });
```