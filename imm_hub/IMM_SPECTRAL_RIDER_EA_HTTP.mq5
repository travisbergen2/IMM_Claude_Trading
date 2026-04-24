//+------------------------------------------------------------------+
//|  IMM_SPECTRAL_RIDER_EA_HTTP.mq5                                  |
//|  SPECTRAL RIDER — Phase II Probe (HTTP Polling Version)          |
//|  Uses WebRequest() — Windows / TradeLocker compatible            |
//|                                                                  |
//|  Author: Travis Bergen | IMM Framework                           |
//|  Zenodo: doi.org/10.5281/zenodo.19075097                         |
//+------------------------------------------------------------------+
#property copyright "Travis Bergen"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>

input string   HubBaseURL     = "https://YOUR_NGROK_URL";
input string   EaId           = "spectral_rider";
input int      PollTimerMs    = 2000;
input int      TickPostInterv = 5;
input int      HttpTimeout    = 3000;
input double   MinLots        = 0.01;
input bool     VerboseLog     = false;

CTrade  g_trade;
string  g_trade_id     = "";
double  g_entry_price  = 0.0;
int     g_direction    = 0;
double  g_lots         = 0.0;
double  g_current_stop = 0.0;
bool    g_in_trade     = false;
int     g_tick_count   = 0;

string  g_tick_url;
string  g_poll_url;
string  g_event_url;
string  g_hb_url;

int OnInit()
{
   g_trade.SetExpertMagicNumber(202502);
   g_trade.SetDeviationInPoints(50);
   g_trade.SetTypeFillingBySymbol(Symbol());

   g_tick_url  = HubBaseURL + "/ea/tick/"      + EaId;
   g_poll_url  = HubBaseURL + "/ea/poll/"      + EaId;
   g_event_url = HubBaseURL + "/ea/event/"     + EaId;
   g_hb_url    = HubBaseURL + "/ea/heartbeat/" + EaId;

   EventSetMillisecondTimer(PollTimerMs);
   DoGet(g_hb_url);
   Print("[SPECTRAL] Ready. Hub=", HubBaseURL);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int r) { EventKillTimer(); }

void OnTick()
{
   g_tick_count++;
   if(g_tick_count % TickPostInterv == 0)
      PostTick();
}

void OnTimer()
{
   PollCommands();
}

void PostTick()
{
   double pnl = 0.0;
   if(g_in_trade)
   {
      ulong t = GetTicket();
      if(t > 0 && PositionSelectByTicket(t))
         pnl = PositionGetDouble(POSITION_PROFIT);
   }

   string body = StringFormat(
      "{\"type\":\"tick_update\","
      "\"symbol\":\"%s\","
      "\"bid\":%.5f,"
      "\"ask\":%.5f,"
      "\"time\":%d,"
      "\"in_trade\":%s,"
      "\"trade_id\":\"%s\","
      "\"open_pnl\":%.2f}",
      Symbol(),
      SymbolInfoDouble(Symbol(), SYMBOL_BID),
      SymbolInfoDouble(Symbol(), SYMBOL_ASK),
      (int)TimeCurrent(),
      g_in_trade ? "true" : "false",
      g_trade_id, pnl
   );
   string r = "";
   DoPost(g_tick_url, body, r);
}

void PollCommands()
{
   string resp = "";
   if(!DoGet(g_poll_url, resp)) return;
   if(StringLen(resp) < 10) return;

   int cs = StringFind(resp, "\"commands\":[");
   if(cs < 0) return;
   cs += 12;
   int ce = StringFind(resp, "]", cs);
   if(ce < 0) return;
   string arr = StringSubstr(resp, cs, ce - cs);
   if(StringLen(arr) < 3) return;

   int pos = 0; int safety = 0;
   while(pos < StringLen(arr) && safety < 20)
   {
      safety++;
      int os = StringFind(arr, "{", pos);
      if(os < 0) break;
      int depth = 1; int oe = os + 1;
      while(oe < StringLen(arr) && depth > 0)
      {
         ushort c = StringGetCharacter(arr, oe);
         if(c == '{') depth++;
         else if(c == '}') depth--;
         oe++;
      }
      ProcessCommand(StringSubstr(arr, os, oe - os));
      pos = oe;
   }
}

void ProcessCommand(const string json)
{
   if(VerboseLog) Print("[SPECTRAL] CMD: ", json);
   string t = Jstr(json, "\"type\"");
   if(t == "ENTER")              HandleProbe(json);
   else if(t == "CLOSE_ALL")     HandleCloseAll(Jstr(json, "\"reason\""));
   else if(t == "CLOSE_PARTIAL") HandlePartial(Jdbl(json, "\"close_fraction\""));
   else if(t == "TRAIL_STOP")    HandleTrail(Jdbl(json, "\"new_stop\""));
   else if(t == "MOVE_TO_BE_PLUS") HandleBE(Jdbl(json, "\"new_stop\""));
}

void HandleProbe(const string json)
{
   if(g_in_trade) return;
   string dir   = Jstr(json, "\"direction\"");
   double lots  = NormLots(Jdbl(json, "\"lots\""));
   int    sl_p  = (int)Jdbl(json, "\"initial_sl_pips\"");
   g_trade_id   = Jstr(json, "\"trade_id\"");

   double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   double digits= (double)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS);
   double pip   = (digits == 5 || digits == 3) ? point * 10 : point;
   int    otype = (dir == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   double ask   = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   double bid   = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   double ep    = (otype == ORDER_TYPE_BUY) ? ask : bid;
   double sl    = (otype == ORDER_TYPE_BUY) ? ep - sl_p * pip : ep + sl_p * pip;
   sl           = NormalizeDouble(sl, (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS));

   bool ok = (otype == ORDER_TYPE_BUY)
      ? g_trade.Buy(lots, Symbol(), 0, sl, 0, "IMM_PROBE")
      : g_trade.Sell(lots, Symbol(), 0, sl, 0, "IMM_PROBE");

   if(ok)
   {
      g_in_trade = true; g_entry_price = ep;
      g_direction = (otype == ORDER_TYPE_BUY) ? 1 : -1;
      g_lots = lots; g_current_stop = sl;
      Print("[SPECTRAL] PROBE ", dir, " ", lots, " @ ", ep, " SL=", sl);

      string ev = StringFormat(
         "{\"type\":\"trade_opened\",\"trade_id\":\"%s\","
         "\"symbol\":\"%s\",\"direction\":\"%s\","
         "\"entry_price\":%.5f,\"lots\":%.2f}",
         g_trade_id, Symbol(), dir, ep, lots);
      string r = ""; DoPost(g_event_url, ev, r);
   }
   else
      Print("[SPECTRAL] Probe FAILED err=", GetLastError());
}

void HandleCloseAll(const string reason)
{
   if(!g_in_trade) return;
   double cp = (g_direction == 1)
      ? SymbolInfoDouble(Symbol(), SYMBOL_BID)
      : SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   bool ok = (g_direction == 1)
      ? g_trade.Sell(g_lots, Symbol(), 0, 0, 0, "IMM_CLOSE")
      : g_trade.Buy(g_lots,  Symbol(), 0, 0, 0, "IMM_CLOSE");
   if(ok)
   {
      string ev = StringFormat(
         "{\"type\":\"trade_closed\",\"trade_id\":\"%s\","
         "\"close_price\":%.5f,\"reason\":\"%s\"}",
         g_trade_id, cp, reason);
      string r = ""; DoPost(g_event_url, ev, r);
      g_in_trade = false; g_trade_id = ""; g_lots = 0;
      Print("[SPECTRAL] CLOSED reason=", reason);
   }
}

void HandlePartial(const double frac)
{
   if(!g_in_trade || frac <= 0) return;
   double cl = NormLots(g_lots * frac);
   if(cl >= g_lots - 0.001) { HandleCloseAll("partial_full"); return; }
   bool ok = (g_direction == 1)
      ? g_trade.Sell(cl, Symbol(), 0, 0, 0, "IMM_PART")
      : g_trade.Buy(cl,  Symbol(), 0, 0, 0, "IMM_PART");
   if(ok) { g_lots -= cl; if(g_lots < MinLots) g_in_trade = false; }
}

void HandleTrail(const double ns)
{
   if(!g_in_trade || ns <= 0) return;
   double pt = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   bool mv = (g_direction == 1) ? ns > g_current_stop + pt : ns < g_current_stop - pt;
   if(!mv) return;
   ulong t = GetTicket();
   if(t > 0 && g_trade.PositionModify(t, ns, 0))
      g_current_stop = ns;
}

void HandleBE(const double bs)
{
   if(!g_in_trade || bs <= 0) return;
   ulong t = GetTicket();
   if(t > 0 && g_trade.PositionModify(t, bs, 0))
      g_current_stop = bs;
}

ulong GetTicket()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(PositionSelectByTicket(t) &&
         PositionGetString(POSITION_SYMBOL) == Symbol() &&
         PositionGetInteger(POSITION_MAGIC) == 202502)
         return t;
   }
   return 0;
}

double NormLots(double l)
{
   double s = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   double mn= SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   double mx= MathMin(SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX), 5.0);
   return MathMax(mn, MathMin(mx, NormalizeDouble(MathRound(l/s)*s, 2)));
}

bool DoPost(const string url, const string body, string &resp)
{
   char d[]; char r[]; string h;
   StringToCharArray(body, d, 0, StringLen(body));
   ArrayResize(d, ArraySize(d) - 1);
   string rh = "Content-Type: application/json\r\n";
   int code = WebRequest("POST", url, rh, HttpTimeout, d, r, h);
   if(code < 0) { if(GetLastError()==4014) Print("[SPECTRAL] Add URL: ",url); return false; }
   resp = CharArrayToString(r); return true;
}

bool DoGet(const string url, string &resp)
{
   char d[]; char r[]; string h;
   ArrayResize(d, 0);
   string rh = "Accept: application/json\r\n";
   int code = WebRequest("GET", url, rh, HttpTimeout, d, r, h);
   if(code < 0) return false;
   resp = CharArrayToString(r); return true;
}

bool DoGet(const string url)
{ string r = ""; return DoGet(url, r); }

// JSON helpers
string Jstr(const string j, const string k)
{
   int s = StringFind(j, k); if(s<0) return "";
   s = StringFind(j,"\"",s+StringLen(k)); if(s<0) return "";
   s++; int e = StringFind(j,"\"",s); if(e<0) return "";
   return StringSubstr(j,s,e-s);
}
double Jdbl(const string j, const string k)
{
   int s = StringFind(j,k); if(s<0) return 0.0;
   s += StringLen(k);
   while(s<StringLen(j) && (StringGetCharacter(j,s)==':'||StringGetCharacter(j,s)==' ')) s++;
   string n=""; int sf=0;
   while(s<StringLen(j)&&sf<30){ushort c=StringGetCharacter(j,s);if(c==','||c=='}'||c==']'||c==' ')break;n+=ShortToString(c);s++;sf++;}
   return StringToDouble(n);
}
