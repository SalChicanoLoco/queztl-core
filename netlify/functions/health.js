exports.handler = async (event, context) => {
    // Allow CORS
    const headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Content-Type': 'application/json'
    };

    if (event.httpMethod === 'OPTIONS') {
        return { statusCode: 200, headers, body: '' };
    }

    return {
        statusCode: 200,
        headers,
        body: JSON.stringify({
            service: "Queztl-Core Testing & Monitoring System",
            status: "running",
            version: "1.0.0",
            note: "Serverless function - Full backend requires separate deployment"
        })
    };
};
