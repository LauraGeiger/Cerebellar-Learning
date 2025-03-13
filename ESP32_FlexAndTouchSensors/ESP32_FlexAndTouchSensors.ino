// Define Pins
const int potPin = 34;  // Analog Input
const int outputPin = 15;  // Digital Output
const int muxChannels[] = {2, 4, 5, 18};  // MUX control pins
const int numFlexSensors = 10;
const int numTouchSensors = 2;
const int numSensors = numFlexSensors + numTouchSensors;
const int numReadings = 10;

const int thresholdFlexSensor = 100;
const int maxTouchSensor = -4095;

int baseline[numSensors];   // To store baseline value
int flexsens[numSensors];

void setup() {
    Serial.begin(115200);
    pinMode(outputPin, OUTPUT);
    digitalWrite(outputPin, HIGH);

    for (int i = 0; i < 4; i++) {
        pinMode(muxChannels[i], OUTPUT);
    }

    delay(1000);  // Wait for stability
    recordBaseline();

    Serial.print("Baseline Values: ");
    for (int i = 0; i < numSensors; i++) {
        Serial.print(baseline[i]);
        if (i < numSensors - 1) Serial.print(", ");
    }
    Serial.println();
}

void loop() {    
    for (int i = 0; i < numSensors; i++) {
        muxWrite(i);
        digitalWrite(outputPin, LOW);
        delay(1);

        int rawValue = analogRead(potPin);
        int normalizedValue = rawValue - baseline[i];
        int value = 0;
        if (i < numFlexSensors) {
            if (normalizedValue > thresholdFlexSensor) {
                value = 1;
            }
        } 
        else {
            value = int(normalizedValue / float(maxTouchSensor) * 100);
        }
        flexsens[i] = value;

        digitalWrite(outputPin, HIGH);
        delay(1);

        // Send sensor values as a CSV line (comma-separated)
        Serial.print(flexsens[i]);
        if (i < numSensors-1) Serial.print(",");  // Add comma except for last value
    }
    Serial.println();  // Newline to mark end of data packet

    delay(100);
}

void muxWrite(int channel) {
    
    for (int i = 0; i < 4; i++) {
        digitalWrite(muxChannels[i], bitRead(channel, i));
    }
}

// Function to record baseline using first 10 sensor readings
void recordBaseline() {
    for (int i = 0; i < numSensors; i++) {
        int sum = 0;
        for (int j = 0; j < numReadings; j++) {
            muxWrite(i);
            digitalWrite(outputPin, LOW);
            delay(1);

            sum += analogRead(potPin);
            digitalWrite(outputPin, HIGH);
            delay(1);
        }
        baseline[i] = int(sum / float(numReadings));  // Compute mean for this sensor
    }
}

