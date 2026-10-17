/*
 ==============================================================================
  AI Fall Detection System - ESP32-CAM Dedicated Video Streamer (Zero-Lag + Auto-Retry)
 ==============================================================================
  Board: AI Thinker ESP32-CAM
  Role: Ultra-Low Latency Wireless Video Feed (400x296 CIF @ 25-30 FPS)
 ==============================================================================
*/

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ======================== 1. WI-FI CONFIGURATION ========================
const char* ssid = "Esp";             // Your Wi-Fi Name
const char* password = "5567heyy";    // Your Wi-Fi Password

// ======================== 2. AI-THINKER CAMERA PINS =====================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

httpd_handle_t stream_httpd = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ======================== MJPEG STREAM HANDLER ========================
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("[CAM] Frame capture failed");
      res = ESP_FAIL;
    } else {
      _jpg_buf_len = fb->len;
      _jpg_buf = fb->buf;
    }

    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, 64, _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }

    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if (_jpg_buf) {
      free(_jpg_buf);
      _jpg_buf = NULL;
    }

    if (res != ESP_OK) break;
  }
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("[HTTP] Camera stream server running on port 80.");
  }
}

// ======================== SETUP & LOOP ===============================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- Starting Fall Detection ESP32-CAM Node ---");

  // Camera Configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Optimized for smooth real-time AI video streaming (25-30 FPS, zero lag)
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_CIF;  // 400x296 (Ultra-fast, crisp for YOLO pose)
    config.jpeg_quality = 14;
    config.fb_count     = 2;
    Serial.println("[INFO] PSRAM detected: Stream set to 400x296 (CIF) for 30 FPS zero-lag.");
  } else {
    config.frame_size   = FRAMESIZE_QVGA; // 320x240
    config.jpeg_quality = 16;
    config.fb_count     = 1;
    Serial.println("[WARN] No PSRAM: Stream set to 320x240 (QVGA).");
  }

  // Initialize Camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init failed: 0x%x\n", err);
    return;
  }
  Serial.println("[CAM] Camera hardware initialized successfully.");

  // Flip sensor so image is right-side up
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_vflip(s, 1);   // 1 = flip vertically
    s->set_hmirror(s, 1); // 1 = flip horizontally
  }

  // Connect to Wi-Fi with Auto-Retry & No Sleep
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // Keeps Wi-Fi radio at maximum power
  WiFi.begin(ssid, password);
  Serial.printf("[WIFI] Connecting to %s ", ssid);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    attempts++;
    if (attempts > 30) { // If not connected after 15 seconds, retry handshake
      Serial.println("\n[WIFI] Retrying handshake...");
      WiFi.disconnect();
      WiFi.begin(ssid, password);
      attempts = 0;
    }
  }

  Serial.println("\n[WIFI] Connected!");
  Serial.print("[INFO] Camera Stream URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");

  // Start HTTP video streamer
  startCameraServer();
}

void loop() {
  delay(1000);
}
