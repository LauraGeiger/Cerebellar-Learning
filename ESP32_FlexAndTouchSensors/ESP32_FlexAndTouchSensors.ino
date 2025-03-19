// Define Pins of MUX1
const int mux1potPin = 35;  // Analog Input
const int mux1outputPin = 19;  // Digital Output
const int mux1Channels[] = {21, 22, 23, 32};  // MUX 1 control pins

// Define Pins of MUX2
const int mux2potPin = 34;  // Analog Input
const int mux2outputPin = 15;  // Digital Output
const int mux2Channels[] = {2, 4, 5, 18};  // MUX 2 control pins

const int numFlexSensors = 10;
const int numTouchSensors = 2;
const int numSoftTouchSensors = 4;
const int numSensors = numFlexSensors + numTouchSensors + numSoftTouchSensors;
const int mux1numSensors = numFlexSensors + numTouchSensors;
const int numReadings = 10;

const int thresholdFlexSensor = 150;
const int maxTouchSensor = 4095;
const int maxSoftTouchSensor = 4095;

int baseline[numSensors];   // To store baseline value
int sensors[numSensors];

void setup() {
    Serial.begin(115200);
    pinMode(mux1outputPin, OUTPUT);
    pinMode(mux2outputPin, OUTPUT);
    digitalWrite(mux1outputPin, HIGH);
    digitalWrite(mux2outputPin, HIGH);

    for (int i = 0; i < 4; i++) {
        pinMode(mux1Channels[i], OUTPUT);
        pinMode(mux2Channels[i], OUTPUT);
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
    for (int s = 0; s < numSensors; s++) {
        int value = 0;

        if (s < mux1numSensors) {
            for (int i = 0; i < 4; i++) {
                digitalWrite(mux1Channels[i], bitRead(s, i));
            }

            digitalWrite(mux1outputPin, LOW);
            delay(1);

            int rawValue = analogRead(mux1potPin);
            
            if (s < numFlexSensors) {
                int normalizedValue = rawValue - baseline[s];
                if (normalizedValue > thresholdFlexSensor) {
                    value = 1;
                }
            } 
            else {
                value = 100 - int(rawValue / float(maxTouchSensor) * 100);
            }

            digitalWrite(mux1outputPin, HIGH);
            delay(1);
        }
        else {
            for (int i = 0; i < 4; i++) {
                digitalWrite(mux2Channels[i], bitRead(s, i));
            }

            digitalWrite(mux2outputPin, LOW);
            delay(1);

            int rawValue = analogRead(mux2potPin);
            int normalizedValue = 100 - int(baseline[s] / float(maxSoftTouchSensor) * 100);
            value = 100 - int(rawValue / float(maxSoftTouchSensor) * 100);
            //value = value - normalizedValue;

            digitalWrite(mux2outputPin, HIGH);
            delay(1);
        }

        sensors[s] = value;

        // Send sensor values as a CSV line (comma-separated)
        Serial.print(sensors[s]);
        if (s < numSensors-1) Serial.print(",");  // Add comma except for last value
    }
    Serial.println();  // Newline to mark end of data packet

    delay(100);
}

// Function to record baseline using first 10 sensor readings
void recordBaseline() {
    for (int s = 0; s < numSensors; s++) {
        int sum = 0;
        
        for (int nr = 0; nr < numReadings; nr++) {
            if (s < mux1numSensors) {
                for (int i = 0; i < 4; i++) {
                    digitalWrite(mux1Channels[i], bitRead(s, i));
                }
                digitalWrite(mux1outputPin, LOW);
                delay(1);

                sum += analogRead(mux1potPin);
                digitalWrite(mux1outputPin, HIGH);
                delay(1);
            }
            else {
                for (int i = 0; i < 4; i++) {
                    digitalWrite(mux2Channels[i], bitRead(s, i));
                }
                digitalWrite(mux2outputPin, LOW);
                delay(1);

                sum += analogRead(mux2potPin);
                digitalWrite(mux2outputPin, HIGH);
                delay(1);
            }
        }
        
        baseline[s] = int(sum / float(numReadings));  // Compute mean for this sensor
    }
}

