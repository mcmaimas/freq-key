# Purpose
Create a custom keyboard based on typing patterns. Keys will grow/shrink based on 
- How often I use them
- What area of the key I press when I type
- How I often I use them in relation to adjacent keys

Example: I type 's' a lot more than 'x' but they are given the same amount of real estate on the keyboard. Typically when I type 's' I hit the bottom half of the key as well. The S key and x key size and shape should reflect this.

# Rules
- The keys should change size based on their frequency
- A key has no maximum size
- A key will have a minimum size of 3mmx5mm
- The total surface area covered by the keys stays the same
- This means when one key grows, it takes up space vacated by another space
- There will always be a fixed amount of padding between the keys 

# Implementation Details
Key Size: 7mm x 10mm - Padding: 1mm - Row Offset: 4mm

Mock Character Frequency Distribution for the English Language
| Q: 0.10% | W: 2.4% | E: 12.7% | R: 6.0% | T: 9.1% | Y: 2.0% | U: 2.8% | I: 7.0% | O: 7.5% | P: 1.9% |
| A: 8.2% | S: 6.3% | D: 4.3% | F: 2.2% | G: 2.0% | H: 6.1% | J: 0.15% | K: 0.8% | L: 4.0% |
| Z: 0.07% | X: 0.15% | C: 2.8% | V: 1.0% | B: 1.5% | N: 6.7% | M: 2.4% |

The initial layout will be based off a distribution of characters and then update slightly based on personal key tracking

