def metric_catalog():
    """
    Chart Builder metric definitions.

    key = dataframe column name
    label = UI display name
    """
    return [
        {"key": "QBVision_Rating", "label": "QBVision Rating"},
        {"key": "next_QBVision_Rating", "label": "Next QBVision Rating (actual)"},
        {"key": "epa_per_db", "label": "EPA per Dropback"},
        {"key": "context_pred_epa_per_db", "label": "Context Pred EPA/DB"},
        {"key": "skill_residual", "label": "Skill Residual (Context Adj)"},
        {"key": "dropbacks", "label": "Dropbacks"},
        {"key": "int_rate", "label": "INT Rate"},
        {"key": "sack_rate", "label": "Sack Rate"},
        {"key": "ay_per_att", "label": "Air Yards per Attempt"},
        {"key": "yac_per_comp", "label": "YAC per Completion"},
        {"key": "team_sack_rate_allowed", "label": "Team Sack Rate Allowed"},
        {"key": "team_yac_per_comp", "label": "Team YAC per Completion"},
        {"key": "team_no_huddle_rate", "label": "Team No-Huddle Rate"},
        {"key": "efficiency_score", "label": "Efficiency Score"},
        {"key": "decision_score", "label": "Decision Score"},
        {"key": "explosiveness_score", "label": "Explosiveness Score"},
        {"key": "mobility_score", "label": "Mobility Score"},
    ]
