import axios from 'axios'

export const getProdText = async (message) => {
    const url = "https://us-central1-dmura-329500.cloudfunctions.net/handler"
    const data = {
        "message": message
    }
    const config = {
        headers: {
            'Content-Type': 'application/json'
        }
    }
    const response = await axios.post(url, data, config)
    return response.data
}