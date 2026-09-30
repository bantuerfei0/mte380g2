/**
 * Since the Raspberry Pi 5 does not have a ADC, we use an ATMega2560 as an ADC
 * Data is sent over serial
 * 2026-09-30
 */

// for mux
const uint8_t MUX_PINS[3] = { 0, 0, 0 };

// mux output
const uint8_t MUX_OUTPUT_PIN = 0;
uint8_t counter = 0;  // keep track of which led we're sending

uint8_t light_value = 0;

void update_mux() {
  for (uint8_t i = 0; i < 3; i++) {
    if (counter & (1 << i)) {
      digitalWrite(MUX_PINS[i], HIGH);
    } else {
      digitalWrite(MUX_PINS[i], LOW);
    }
  }
}

void setup() {
  for (uint8_t i = 0; i < 3; i++) pinMode(MUX_PINS[i], OUTPUT);
  // 8 data 1 stop (8N1) by default
  Serial.begin(115200L);  // 38.4k-ish maximum, will increase if too slow
  while (!Serial) {}
}

void loop() {
  update_mux(); // switch mux to correct pin
  // perhaps needs a pause here? mux takes nanoseconds to switch though. See: https://www.ti.com/lit/ds/scls542c/scls542c.pdf?ts=1790803658641
  light_value = analogRead(MUX_OUTPUT_PIN) >> 2; // shift out last 2 bits

  Serial.write(0x80 | (counter << 4) | (light_value >> 7)); // MSB=1 for first
  Serial.write(light_value & 0x7F); // MSB=0 for second packet
  // Serial.flush(); // potentially not needed, see if get timing issues
  counter = (counter + 1) & 0x07;
}
