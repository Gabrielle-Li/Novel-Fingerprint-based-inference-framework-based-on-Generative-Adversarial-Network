# Novel-Fingerprint-based-inference-framework-based-on-Generative-Adversarial-Network

As indoor localization becomes a necessity to provide intelligent location-based services for in-building users, fingerprint-based positioning has been widely adopted in a number of Wi-Fi-equipped devices. However, this approach requires an extensive prior site survey and thus can not be applied to unexplored environments where any prior fingerprint sampling has not been conducted. To address the problem, we propose a novel fingerprint-based inference framework based on a Generative Adversarial Network (GAN) by extracting the underlying correlation between a location coordinate and its radio signal features. This work empowers indoor localization in unknown areas, including unknown data points, newly deployed APs, or unexplored sites, by 1) decomposing into a signal feature map for each AP; 2) processing learning with a set of location and its associated signal strength; 3) generating and integrating synthetic radio fingerprints; and 4) employing them into some existing localization algorithms. We evaluated GAN- Loc with extensive real-world RSSI experiments in seven different real-world indoor places across various wireless radios.

We propose three problems in this project. The first one is to set up two different APs in one known place. We randomly walk in the room and get different Received Signal Strength in different location to two separate APs. Then we use GAN to generate the localization fingerprint map of the whole room. 
The second one is to set up one APs in one known room and also get different RSS in our random path. Then we use GAN to generate different fingerprint maps from different APs. 
The third one is we get the RSS from a known area and its fingerprint map from one settles AP and generate other fingerprint map in other unknown area. 

The technical report is uploaded in the file named 'Report'.
