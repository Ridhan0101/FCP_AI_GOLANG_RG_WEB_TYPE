package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

// AIModelConnector struct to store http.Client
type AIModelConnector struct {
	Client *http.Client
}

// Inputs struct to define the format of input for the AI model
type Inputs struct {
	Table map[string][]string `json:"table"`
	Query string              `json:"query"`
}

// Response struct to define the format of response from the AI model
type Response struct {
	Answer      string   `json:"answer"`
	Coordinates [][]int  `json:"coordinates"`
	Cells       []string `json:"cells"`
	Aggregator  string   `json:"aggregator"`
}

// CsvToSlice function to convert CSV into a map
func CsvToSlice(data string) (map[string][]string, error) {
	reader := csv.NewReader(strings.NewReader(data))
	records, err := reader.ReadAll() // Read all data from CSV
	if err != nil {
		return nil, err
	}

	if len(records) < 1 {
		return nil, errors.New("no data found")
	}

	header := records[0]
	result := make(map[string][]string)

	for i, col := range header {
		result[col] = make([]string, 0, len(records)-1)
		for _, record := range records[1:] {
			if i < len(record) {
				result[col] = append(result[col], record[i])
			}
		}
	}

	return result, nil
}

// ConnectAIModel function to connect to the AI model and get the response
func (c *AIModelConnector) ConnectAIModel(payload Inputs, token string) (Response, error) {
	url := "https://api-inference.huggingface.co/models/google/tapas-base-finetuned-wtq"
	data, err := json.Marshal(payload) // Convert payload to JSON
	if err != nil {
		return Response{}, err
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(data))
	if err != nil {
		return Response{}, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	// Retry logic to attempt connecting to the AI model if it fails
	maxRetries := 10
	for i := 0; i < maxRetries; i++ {
		resp, err := c.Client.Do(req)
		if err != nil {
			return Response{}, err
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			var aiResponse Response
			if err := json.NewDecoder(resp.Body).Decode(&aiResponse); err != nil {
				return Response{}, err
			}
			return aiResponse, nil
		}

		if resp.StatusCode == http.StatusServiceUnavailable {
			var result map[string]interface{}
			body, _ := ioutil.ReadAll(resp.Body)
			if err := json.Unmarshal(body, &result); err == nil {
				if estimatedTime, ok := result["estimated_time"].(float64); ok {
					log.Printf("Model is currently loading, retrying in %.1f seconds...\n", estimatedTime)
					time.Sleep(time.Duration(estimatedTime) * time.Second)
					continue
				}
			}
		}

		body, _ := ioutil.ReadAll(resp.Body)
		return Response{}, fmt.Errorf("failed to connect to AI model, status: %s, response: %s", resp.Status, string(body))
	}

	return Response{}, fmt.Errorf("max retries reached, failed to connect to AI model")
}

func main() {
	// Load environment variables from .env file
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file: %v\n", err)
	}

	// Get Huggingface API token from environment variables
	token := os.Getenv("HUGGINGFACE_TOKEN")
	if token == "" {
		log.Fatalf("HUGGINGFACE_TOKEN not found in .env file")
	}

	// Path to CSV file
	csvFile := "data-series.csv"

	// Read CSV file
	data, err := ioutil.ReadFile(csvFile)
	if err != nil {
		log.Fatalf("Error reading CSV file: %v\n", err)
	}

	// Parse CSV to slice
	table, err := CsvToSlice(string(data))
	if err != nil {
		log.Fatalf("Error parsing CSV file: %v\n", err)
	}

	// Create AI model connector
	client := &http.Client{}
	connector := &AIModelConnector{Client: client}

	// Set up Gin router
	r := gin.Default()

	// Serve static files
	r.Static("/static", "./static")

	// Load HTML templates
	r.LoadHTMLGlob("templates/*")

	// Serve the home page
	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "home.html", gin.H{})
	})

	// Serve the chatbot page
	r.GET("/chatbot", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", gin.H{})
	})

	// Serve the contact page
	r.GET("/contact", func(c *gin.Context) {
		c.HTML(http.StatusOK, "contact.html", gin.H{})
	})

	// Handle chatbot query
	r.POST("/query", func(c *gin.Context) {
		query := c.PostForm("query")
		if query == "" {
			c.HTML(http.StatusBadRequest, "index.html", gin.H{"error": "Query cannot be empty"})
			return
		}

		payload := Inputs{
			Table: table,
			Query: query,
		}

		response, err := connector.ConnectAIModel(payload, token)
		if err != nil {
			c.HTML(http.StatusInternalServerError, "index.html", gin.H{"error": fmt.Sprintf("Error connecting to AI model: %v", err)})
			return
		}

		// Display response
		c.HTML(http.StatusOK, "index.html", gin.H{
			"query":       query,
			"answer":      response.Answer,
			"coordinates": response.Coordinates,
			"cells":       response.Cells,
			"aggregator":  response.Aggregator,
		})
	})

	// Start the server
	r.Run(":8080")
}
