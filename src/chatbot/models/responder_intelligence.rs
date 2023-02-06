pub trait ResponderIntelligence {
    fn new() -> Self;
    fn respond(&self, request: &str) -> String;
}
