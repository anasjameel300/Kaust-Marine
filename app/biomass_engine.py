"""
Scientific Calibration & FishBase Allometric Biomass Engine (Option 1)
KAUST Marine Vision - Research Grade Ecological Analytics

Translates sub-pixel oriented bounding boxes (snout-to-tail length and body depth) into:
1. Physical Metric Fork Length (cm) via 2-Point Reference Calibration / Preset Rigs
2. Species/Family-Specific Wet-Weight Biomass: W = a * (L ^ b) [FishBase Parameters]
3. Ecological Stock Assessment Metrics:
   - Total Biomass (grams & kg)
   - Commercial Landing Yield (kg & %)
   - Community Shannon-Wiener Biodiversity Index (H')
   - Trophic Functional Guild Allocation
"""

import math
from typing import Dict, List, Any

# Standard Calibration Rig Presets
CALIBRATION_PRESETS = {
    "bruvs_rig": {
        "name": "In-Situ BRUVs Survey (Dr. Renato Morais)",
        "default_px_per_cm": 22.0,
        "description": "Standard bait canister diameter (8.0 cm) & 1.2m rigid bait arm baseline"
    },
    "measuring_board": {
        "name": "Artisanal Catch Tray (Dr. Jessica Zamborain-Mason)",
        "default_px_per_cm": 35.0,
        "description": "Landed catch on graduated measuring board or standardized fish market tray"
    },
    "custom": {
        "name": "Custom Reference Calibration",
        "default_px_per_cm": 25.0,
        "description": "User-defined scale calibration"
    }
}

class BiomassEngine:
    @staticmethod
    def calculate_individual_morphometrics(
        oriented_length_px: float,
        body_depth_px: float,
        pixels_per_cm: float,
        fb_a: float,
        fb_b: float
    ) -> Dict[str, Any]:
        """
        Calculates metric fork length (cm), body depth (cm), and FishBase biomass weight (g).
        """
        px_per_cm = max(1.0, float(pixels_per_cm))
        length_cm = round(float(oriented_length_px) / px_per_cm, 2)
        depth_cm = round(float(body_depth_px) / px_per_cm, 2)

        # Allometric length-weight equation: W = a * (L ^ b)
        # Length in cm -> Weight in grams
        weight_grams = round(float(fb_a) * math.pow(max(0.1, length_cm), float(fb_b)), 1)
        weight_kg = round(weight_grams / 1000.0, 3)

        aspect_ratio = round(length_cm / max(0.01, depth_cm), 2)

        return {
            "length_cm": length_cm,
            "fork_length_cm": length_cm,
            "depth_cm": depth_cm,
            "body_depth_cm": depth_cm,
            "weight_grams": weight_grams,
            "biomass_g": weight_grams,
            "weight_kg": weight_kg,
            "aspect_ratio": aspect_ratio
        }

    @staticmethod
    def calculate_community_metrics(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes community-level ecological biomass, stock yield, and Shannon diversity.
        """
        fish_detections = [d for d in detections if d.get("class_name") == "fish"]
        total_fish = len(fish_detections)

        if total_fish == 0:
            return {
                "total_fish_count": 0,
                "total_biomass_grams": 0.0,
                "total_biomass_kg": 0.0,
                "commercial_biomass_kg": 0.0,
                "commercial_yield_pct": 0.0,
                "shannon_diversity_h": 0.0,
                "family_distribution": {},
                "guild_distribution": {}
            }

        total_weight_grams = sum(d.get("biomass", {}).get("weight_grams", 0.0) for d in fish_detections)
        commercial_weight_grams = sum(
            d.get("biomass", {}).get("weight_grams", 0.0)
            for d in fish_detections
            if d.get("taxonomy", {}).get("is_commercial", False)
        )

        total_biomass_kg = round(total_weight_grams / 1000.0, 3)
        commercial_biomass_kg = round(commercial_weight_grams / 1000.0, 3)
        commercial_pct = round((commercial_weight_grams / max(1e-3, total_weight_grams)) * 100.0, 1)

        # Family distribution & Shannon-Wiener Diversity Index: H' = - sum(p_i * ln(p_i))
        family_counts: Dict[str, int] = {}
        guild_counts: Dict[str, int] = {}

        for d in fish_detections:
            fam = d.get("taxonomy", {}).get("family", "Unknown")
            guild = d.get("taxonomy", {}).get("trophic_guild", "General")
            family_counts[fam] = family_counts.get(fam, 0) + 1
            guild_counts[guild] = guild_counts.get(guild, 0) + 1

        shannon_h = 0.0
        for count in family_counts.values():
            p_i = count / total_fish
            if p_i > 0:
                shannon_h -= p_i * math.log(p_i)

        return {
            "total_fish_count": total_fish,
            "total_biomass_grams": round(total_weight_grams, 1),
            "total_biomass_kg": total_biomass_kg,
            "commercial_biomass_kg": commercial_biomass_kg,
            "commercial_yield_pct": commercial_pct,
            "commercial_ratio_pct": commercial_pct,
            "shannon_diversity_h": round(shannon_h, 3),
            "shannon_diversity_index": round(shannon_h, 3),
            "family_distribution": family_counts,
            "guild_distribution": guild_counts
        }

# Global singleton
biomass_engine = BiomassEngine()
