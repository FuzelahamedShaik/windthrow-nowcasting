## Text-Augmented Windthrow Nowcasting with a Multi-Agent Automation Framework

### University of Oulu - Centre for Machine Vision & Signal Processing (CMVS)

## Overview

This repository contains the full implementation of a multi-agent, text-augmented nowcasting system for predicting storm-induced windthrow risk in Finnish forests at a 1 km resolution. The system integrates:

- Numerical weather forecasts (FMI) 
- Forest structural data (Metsäkeskus)
- Terrain/topography
- CAP weather warning texts (XML)
- Remote sensing-based and EMS ground-truth damage labels

A multi-agent system (MAS) architecture orchestrates automated:

- Data ingestion
- Text processing
- Feature fusion
- Probabilistic modeling
- Uncertainty quantification
- Explainable AI summaries
- Near-real-time monitoring and inference

This project supports a reproducible research pipeline and operational early-warning experiments for Finnish forestry, emergency management, and energy resilience.

## Data sources

All data sources used are openly available:

- FMI Open Data API - forecasts, observations
- FMI CAP XML Warnings - textual severe weather bulletins
- Metsäkeskus Open Forest Data - forest stand features
- NLS DEM 10m - terrain data
- Sentinel-2 - pre/post-strom NDVI/NDMI
- Copernicus EMS - storm damage ground truth
- LUKE - regional storm damage assessments

## Multi-Agent System (MAS) Architecture Diagram

![Multi-Agent System Architecture](assets/diagrams/mas_architecture.png)