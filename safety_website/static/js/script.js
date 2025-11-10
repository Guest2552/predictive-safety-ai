/* --- static/js/script.js --- */

document.addEventListener("DOMContentLoaded", () => {
    
    // --- 1. Scroll-In Animation Observer ---
    
    // Find all elements to fade in
    const fadeElements = document.querySelectorAll('.fade-in');

    const observerOptions = {
        root: null, // relative to the viewport
        rootMargin: '0px',
        threshold: 0.1 // 10% of the item must be visible
    };

    // Create the observer
    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Add the 'is-visible' class to trigger the animation
                entry.target.classList.add('is-visible');
                // Stop observing the element so it only fades in once
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe each fade-in element
    fadeElements.forEach(el => {
        observer.observe(el);
    });

    
    // --- 2. Demo Page Functionality ---
    
    const predictForm = document.getElementById('predict-form');
    
    // Only run this code if we are on the demo page
    if (predictForm) {
        const reportText = document.getElementById('report-text');
        const predictButton = document.getElementById('predict-button');
        
        // Result elements
        const errorMsg = document.getElementById('error-message');
        const resultPlaceholder = document.getElementById('result-box-placeholder');
        const resultContent = document.getElementById('result-content');
        const predictionOutput = document.getElementById('prediction-output');
        const confidenceOutput = document.getElementById('confidence-output');
        const processedTextOutput = document.getElementById('processed-text-output');
        const llmOutput = document.getElementById('llm-output'); // <-- NEW ELEMENT
        
        // Example list
        const exampleList = document.getElementById('example-list');

        // Handle form submission
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault(); // Stop the page from reloading
            
            const text = reportText.value.trim();
            if (!text) {
                showError('Please enter an incident report.');
                return;
            }

            // Disable button and show loading state
            predictButton.disabled = true;
            predictButton.textContent = 'Analyzing...';
            clearResult();

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ report_text: text })
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.error || 'A server error occurred.');
                }

                const data = await response.json();
                showResult(data);

            } catch (error) {
                showError(error.message);
            } finally {
                // Re-enable the button
                predictButton.disabled = false;
                predictButton.textContent = 'Analyze Risk';
            }
        });

        // Function to show a successful result
        function showResult(data) {
            // Show local model prediction
            predictionOutput.textContent = data.prediction;
            predictionOutput.setAttribute('data-risk', data.prediction);
            confidenceOutput.textContent = (data.confidence * 100).toFixed(1);
            
            // Show processed text
            processedTextOutput.textContent = data.processed_text || 'N/A';
            
            // Show LLM analysis
            llmOutput.textContent = data.llm_analysis || 'No analysis provided.';
            
            // Toggle visibility
            resultPlaceholder.classList.add('hidden');
            resultContent.classList.remove('hidden');
        }
        
        // Function to show an error message
        function showError(message) {
            errorMsg.textContent = message;
        }

        // Function to clear previous results and errors
        function clearResult() {
            errorMsg.textContent = '';
            resultPlaceholder.classList.remove('hidden');
            resultContent.classList.add('hidden');
            
            predictionOutput.textContent = '--';
            predictionOutput.removeAttribute('data-risk');
            confidenceOutput.textContent = '--';
            processedTextOutput.textContent = '--';
            llmOutput.textContent = '--'; // Clear LLM output
        }

        // Handle clicking on examples
        if (exampleList) {
            exampleList.addEventListener('click', (e) => {
                if (e.target.tagName === 'LI') {
                    // Copy example text into the textarea
                    reportText.value = e.target.textContent;
                    // Scroll to the top of the form
                    predictForm.scrollIntoView({ behavior: 'smooth' });
                    // Clear any old results
                    clearResult();
                }
            });
        }
    }
});