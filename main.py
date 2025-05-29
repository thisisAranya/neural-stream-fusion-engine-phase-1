if __name__ == "__main__":
    import uvicorn
    from config.settings import settings
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.system_mode == "development",
        log_level=settings.log_level.lower()
    )
