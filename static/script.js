document.addEventListener('DOMContentLoaded', function() {
    const inputSentence = document.getElementById('inputSentence');
    const modelSelect = document.getElementById('modelSelect');
    const predictButton = document.getElementById('predictButton');
    const predictionResult = document.getElementById('predictionResult');

    predictButton.addEventListener('click', function() {
        const sentence = inputSentence.value;
        const selectedModel = modelSelect.value;
        
        // Creating a request object to send to the backend API
        // Just replace '/api/predict' with the actual API endpoint URL
        console.log(selectedModel)
        var apiUrl = ""

        if(selectedModel === "encdecV1") {
            console.log(selectedModel)
            apiUrl = "/getTranslation/v1/"+ sentence;
            console.log(apiUrl)
        } else if (selectedModel === "encdecV2") {
            console.log(selectedModel)
            apiUrl = "/getTransformerV2/"+ sentence
            console.log(apiUrl)

        } else if (selectedModel === "transformerV1") {
            console.log(selectedModel)
            apiUrl = "/getTransformerV1/"+ sentence
            console.log(apiUrl)
            

        } else if (selectedModel === "transformerV2") {
            console.log(selectedModel)
            apiUrl = "/getTransformerV2/"+ sentence
            console.log(apiUrl)

        }   else if (selectedModel === "transformerV3") {
            console.log(selectedModel)
            apiUrl = "/getTransformerV3/"+ sentence
            console.log(apiUrl)

        }   else if (selectedModel === "transformerV4") {
            console.log(selectedModel)
            apiUrl = "/getTransformerV4/"+ sentence
            console.log(apiUrl)
            

        }   else {
            console.log("Wronfg Selection")
            apiUrl = ''
        }



        const request = new Request(apiUrl, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        fetch(request)
            .then(response => response.json())
            .then(data => {
                // handle the reesponse data here
                predictionResult.innerHTML = `<h3>Translated Sentence :<h3><br> <b>${data['Translated Sentence']}</b>`;
                predictionResult.classList.add('show');
            })
            .catch(error => {
                predictionResult.textContent = 'Error: Unable to fetch prediction.';
                console.error(error);
            });
    });
});
