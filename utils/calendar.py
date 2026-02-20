"""
utils/calendar.py
NYSE calendar utilities:
  - Next trading day for signal display
  - Market open check
  - EST time helper
"""

from datetime import datetime, timedelta
import pytz

try:
    import pandas_market_calendars as mcal
    NYSE_CAL_AVAILABLE = True
except ImportError:
    NYSE_CAL_AVAILABLE = False


def get_est_time() -> datetime:
    """Return current datetime in US/Eastern timezone."""
    return datetime.now(pytz.timezone("US/Eastern"))


def is_market_open_today() -> bool:
    """Return True if today is a NYSE trading day."""
    today = get_est_time().date()
    if NYSE_CAL_AVAILABLE:
        try:
            nyse     = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=today, end_date=today)
            return len(schedule) > 0
        except Exception:
            pass
    return today.weekday() < 5


def get_next_signal_date() -> datetime.date:
    """
    Determine the date for which the model's signal applies.

    Rules:
      - If today is a NYSE trading day AND it is before 09:30 EST
        → signal applies to TODAY (market hasn't opened yet)
      - Otherwise
        → signal applies to the NEXT NYSE trading day
    """
    now_est = get_est_time()
    today   = now_est.date()

    market_not_open_yet = (
        now_est.hour < 9 or
        (now_est.hour == 9 and now_est.minute < 30)
    )

    if NYSE_CAL_AVAILABLE:
        try:
            nyse     = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(
                start_date=today,
                end_date=today + timedelta(days=10),
            )
            if len(schedule) == 0:
                return today   # fallback

            first_day = schedule.index[0].date()

            # Today is a trading day and market hasn't opened → today
            if first_day == today and market_not_open_yet:
                return today

            # Otherwise find first trading day strictly after today
            for ts in schedule.index:
                d = ts.date()
                if d > today:
                    return d

            return schedule.index[-1].date()
        except Exception:
            pass

    # Fallback: simple weekend skip
    candidate = today if market_not_open_yet else today + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def is_sync_window() -> bool:
    """True if current EST time is in the 07:00-08:00 or 19:00-20:00 window."""
    now = get_est_time()
    return (7 <= now.hour < 8) or (19 <= now.hour < 20)
