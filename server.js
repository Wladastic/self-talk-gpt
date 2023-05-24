const express = require('express');
const OpenAI = require('openai');

const app = express();
app.use(express.json());
app.use(express.static('public')); // Serve static files from the public directory

let openai;

app.post('/api-key', (req, res) => {
    const apiKey = req.body.apiKey;
    if (!apiKey) {
        return res.status(400).send('API Key is required');
    }
    openai = new OpenAI(apiKey);
    res.send('API Key received. You can now make requests to /chat');
});

app.post('/chat', async (req, res) => {
    if (!openai) {
        return res.status(400).send('API Key is not set. Send a POST request to /api-key with your API Key');
    }
    const message = req.body.message;
    if (!message) {
        return res.status(400).send('Message is required');
    }
    try {
        const response = await openai.complete({
            engine: 'text-davinci-002',
            prompt: message,
            max_tokens: 60
        });
        res.send(response.data.choices[0].text.trim());
    } catch (err) {
        res.status(500).send(err.message);
    }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Server running on port ${port}`));
